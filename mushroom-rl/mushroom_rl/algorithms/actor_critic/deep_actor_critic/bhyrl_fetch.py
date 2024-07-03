import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.core.logger.console_logger import ConsoleLogger

from copy import deepcopy
from itertools import chain

# BHyRL: Boosted Hybrid RL. Action space is hybrid (A sequential discrete approximator takes as input the continous action and outputs the discrete part of the action)
# Boosting idea is from BCRL (Boosted Curriculum Reinforcement Learning) as introduced in https://openreview.net/pdf?id=anbBFlX1tJ1

class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)
    
    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)


class BHyRLPolicy(Policy):
    """
    The policy is a Gaussian policy squashed by a tanh.

    """
    def __init__(self, mu_approximator, sigma_approximator, discrete_approximator,
                 min_a, max_a, log_std_min, log_std_max, temperature, gauss_noise_cov):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in a given
                state;
            sigma_approximator (Regressor): a regressor computing the variance
                in a given state;
            discrete_approximator (Regressor): a regressor computing the discrete
                action disctribution in a given state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std;
            temperature ([float]): temperature for the Gumbel Softmax;
            gauss_noise_cov ([float]): Add gaussian noise to the drawn actions (if calling 'draw_noisy_action()')

        """
        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator
        self._discrete_approximator = discrete_approximator
        
        self._temperature = torch.tensor(temperature)
        self._gauss_noise_cov = np.array(gauss_noise_cov)
        self._max_a = max_a[:mu_approximator.output_shape[0]]
        self._min_a = min_a[:mu_approximator.output_shape[0]]
        self._delta_a = to_float_tensor(.5 * (self._max_a - self._min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (self._max_a + self._min_a), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        use_cuda = self._mu_approximator.model.use_cuda

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _discrete_approximator='mushroom',
            _max_a='numpy',
            _min_a='numpy',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive',
            _temperature='torch',
            _gauss_noise_cov='numpy'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def draw_noisy_action(self, state):
        # Add clipped gaussian noise (only to the continuous actions!)
        cont_noise = np.random.multivariate_normal(np.zeros(self._mu_approximator.output_shape[0]),np.eye(self._mu_approximator.output_shape[0])*self._gauss_noise_cov)
        noise = np.hstack((cont_noise,np.zeros(self._discrete_approximator.output_shape[0])))
        return np.clip(self.compute_action_and_log_prob_t(state, compute_log_prob=False).detach().cpu().numpy() + noise, np.hstack((self._min_a,np.zeros(self._discrete_approximator.output_shape[0]))), np.hstack((self._max_a,np.ones(self._discrete_approximator.output_shape[0]))))

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        # Continuous
        cont_dist = self.cont_distribution(state)
        a_cont_raw = cont_dist.rsample()
        a_cont = torch.tanh(a_cont_raw)
        a_cont_true = a_cont * self._delta_a + self._central_a

        # Discrete
        # NOTE: Discrete approximator takes both state and continuous action as input (sequential policy)
        discrete_dist = self.discrete_distribution(state, a_cont_true.detach()) # detach to avoid gradients of continuous through here
        a_discrete = discrete_dist.rsample()

        if compute_log_prob:
            # Continuous
            log_prob_cont = cont_dist.log_prob(a_cont_raw).sum(dim=1)
            log_prob_cont -= torch.log(1. - a_cont.pow(2) + self._eps_log_prob).sum(dim=1)
            # Discrete
            log_prob_discrete = discrete_dist.log_prob(a_discrete)
            return torch.hstack((a_cont_true, a_discrete)), log_prob_cont+log_prob_discrete
        else:
            return torch.hstack((a_cont_true, a_discrete))

    def cont_distribution(self, state):
        """
        Compute the continous (Gaussian) policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu = self._mu_approximator.predict(state, output_tensor=True)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        return torch.distributions.Normal(mu, log_sigma.exp())

    def discrete_distribution(self, state, a_cont):
        """
        Compute the discrete policy distribution (categorical) in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.
            a_cont (torch tensor): the set of continuous actions, conditioned on 
                which, the discrete distribution is computed.

        Returns:
            The torch distribution for the provided states.

        """
        if isinstance(state, np.ndarray):
            if self._mu_approximator.model.use_cuda:
                state = torch.from_numpy(state).cuda()
            else:
                state = torch.from_numpy(state)
        logits = self._discrete_approximator.predict(torch.hstack((state, a_cont)), output_tensor=True)

        return GumbelSoftmax(temperature=self._temperature, logits=logits)

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """
        # Continuous dist and action
        cont_distr = self.cont_distribution(state)
        act_cont_raw = cont_distr.rsample()
        act_cont_true = torch.tanh(act_cont_raw) * self._delta_a + self._central_a

        # return sum of cont and discrete entropy
        return torch.mean(cont_distr.entropy()).detach().cpu().numpy().item() + torch.mean(self.discrete_distribution(state, act_cont_true).entropy()).detach().cpu().numpy().item()

    def reset(self):
        pass

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:self._mu_approximator.weights_size+self._sigma_approximator.weights_size]
        discrete_weights = weights[self._mu_approximator.weights_size+self._sigma_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)
        self._discrete_approximator.set_weights(discrete_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()
        discrete_weights = self._discrete_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights, discrete_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._mu_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return chain(self._mu_approximator.model.network.parameters(),
                     self._sigma_approximator.model.network.parameters(),
                     self._discrete_approximator.model.network.parameters())


class BHyRL(DeepAC):
    """
    BHyRL with a Hybrid action space (A sequential discrete approximator takes as input the 
    continous action and outputs the discrete part of the action)

    """
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_discrete_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, log_std_min=-20, log_std_max=2, temperature=1.0, use_entropy=False, target_entropy=None, 
                 gauss_noise_cov=0.0, prior_agents=None, mdp_get_prior_state_fn=None, use_kl_on_q=False, kl_on_q_alpha=1e-3,
                 use_kl_on_pi=False, kl_on_pi_alpha=1e-3, critic_fit_params=None):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigma
                approximator to build;
            actor_discrete_params (dict): parameters of the actor discrete distribution
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            temperature (float): the temperature for the softmax part of the gumbel reparametrization
            use_entropy (bool): Add entropy loss similar to SAC
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            gauss_noise_cov ([float, Parameter]): Add gaussian noise to the drawn actions (if calling 'draw_noisy_action()');
            prior_agents ([mushroom object]): The agent object from agents trained on prior tasks;
            mdp_get_prior_state_fn (mushroom object): We need the mdp get state function for state-specific filtering to use the prior agents;
            use_kl_on_q (bool): Whether to use a kl between the prior task policy and the new policy as a reward
            kl_on_q_alpha (float): Alpha parameter to weight the KL divergence reward
            use_kl_on_pi (bool): Whether to use a kl between the prior task policy and the new policy as a loss on the policy
            kl_on_pi_alpha (float): Alpha parameter to weight the KL divergence loss on the policy
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._freeze_data = False # Flag to fit critic and policy only and not use any new data
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)            
        else:
            self._target_entropy = target_entropy

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)
        
        self._prior_critic_approximators = list()
        self._prior_policies = list()
        if(prior_agents is not None):
            for prior_agent in prior_agents:
                self._prior_critic_approximators.append(prior_agent._target_critic_approximator) # The target_critic_approximator object from agents trained on prior tasks
                self._prior_policies.append(prior_agent.policy) # The policy object from an agent trained on a prior task
        
        self._mdp_get_prior_state_fn = mdp_get_prior_state_fn # The mdp function for state-specific filtering to use the prior agents

        self._use_kl_on_q = use_kl_on_q # Whether to use a kl between the prior task policy and the new policy as a reward
        self._kl_on_q_alpha = kl_on_q_alpha # Alpha parameter to weight the KL divergence reward
        self._use_kl_on_pi = use_kl_on_pi # Whether to use a kl between the prior task policy and the new policy as a loss for the new policy
        self._kl_on_pi_alpha = kl_on_pi_alpha # Alpha parameter to weight the KL divergence loss on the policy
        self._kl_with_prior = np.array([0.0]) # KL divergence with previous policy (numpy)
        self._kl_with_prior_t = torch.tensor(0.0) # KL divergence with previous policy (torch)

        self._use_entropy = use_entropy

        actor_mu_approximator = Regressor(TorchApproximator,
                                          **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator,
                                             **actor_sigma_params)
        actor_discrete_approximator = Regressor(TorchApproximator,
                                             **actor_discrete_params)
        self._actor_last_loss = None # Store actor loss for logging
        
        policy = BHyRLPolicy(actor_mu_approximator,
                           actor_sigma_approximator,
                           actor_discrete_approximator,
                           mdp_info.action_space.low,
                           mdp_info.action_space.high,
                           log_std_min,
                           log_std_max,
                           temperature,
                           gauss_noise_cov)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters(),
                                  actor_discrete_approximator.model.network.parameters())

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch'
        )
        self.console_logger = ConsoleLogger(log_name='')
        self.console_logger.info("=======调用bhyrl_fetch函数部分======")

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        if not(self._freeze_data): # flag to fit only and not use any new data
            self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            if self._use_kl_on_q or self._use_kl_on_pi:
                # Calculate KL divergence between current policy and previous policy
                weights, prior_state =  self._mdp_get_prior_state_fn(state, ik_task=(len(self._prior_policies)==1), reach_task=(len(self._prior_policies)==2)) # prior task policy is from an IK task
                weights[weights < 1.0] = 0.0 # Only use prior task weights that are >= 1
                # Note that policies are not residuals so we only need the KL between the immediate previous task and current task
                prior_cont_dist = self._prior_policies[-1].cont_distribution(prior_state) # use prior_state and weights for the immediate previous task                
                curr_cont_dist = self.policy.cont_distribution(state)
                # Convert to MultivariateNormal distributions (for KL calculation)
                prior_multiv_cont_dist = torch.distributions.MultivariateNormal(prior_cont_dist.mean, torch.diag_embed(prior_cont_dist.variance))
                curr_multiv_cont_dist = torch.distributions.MultivariateNormal(curr_cont_dist.mean, torch.diag_embed(curr_cont_dist.variance))
                # TODO: Add discrete discrete distribution for KL calculation
                # Use Forward KL instead of reverse KL because prior policy distribution could be peaky
                self._kl_with_prior_t = torch.tensor(weights, device=prior_cont_dist.mean.device)*torch.distributions.kl.kl_divergence(prior_multiv_cont_dist,curr_multiv_cont_dist)
                self._kl_with_prior = self._kl_with_prior_t.detach().cpu().numpy()

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                if self._use_entropy:
                    self._update_alpha(log_prob.detach())
                self._actor_last_loss = loss.detach().cpu().numpy() # Store actor loss for logging

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            rho = q # residual q
            for idx, prior_critic in enumerate(self._prior_critic_approximators):
                # # Fitting a 'residual q' i.e 'rho'. So we subtract the prior_rho values
                # Use weights for the prior rho values. Also use appropriate state-spaces as per the prior task
                weights, prior_state =  self._mdp_get_prior_state_fn(state, ik_task=(idx==0), reach_task=(idx==1)) # task[0] is an IK prior task
                rho_prior = weights*prior_critic.predict(prior_state, action, prediction='min')
                rho -= rho_prior # subtract the prior_rho value
            
            if self._use_kl_on_q: # Use KL divergence between the previous task policy and new policy as reward
                rho -= self._kl_on_q_alpha*np.clip(self._kl_with_prior, 0.0, 5000.0) # TWEAK: Clip the KL because it can explode


            self._critic_approximator.fit(state, action, rho,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def _loss(self, state, action_new, log_prob):
        rho_0 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=0)
        rho_1 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=1)

        q = torch.min(rho_0, rho_1)

        for idx, prior_critic in enumerate(self._prior_critic_approximators):
            # # For policy loss, use q value as a combination of prior task and current task residual_q (rho) values
            # Use weights for the prior rho values. Also use appropriate state-spaces as per the prior task
            weights, prior_state =  self._mdp_get_prior_state_fn(state, ik_task=(idx==0), reach_task=(idx==1)) # task[0] is an IK prior task
            weights = torch.tensor(weights, device=q.device)
            # NOTE: We will use the q_prior gradient here for policy fitting but not in the residual_q fitting
            rho_prior = weights * prior_critic.predict(prior_state, action_new, output_tensor=True, prediction='min').values
            q += rho_prior
        
        if self._use_kl_on_pi:
            # Add a KL penalty for deviating from previous policy (with gradients)
            q -= torch.tensor(self._kl_on_pi_alpha, device=q.device)*torch.clip(self._kl_with_prior_t, 0.0, 5000.0) # TWEAK: Clip the KL because it can explode
        
        if self._use_entropy:
            q -= self._alpha * log_prob

        return  -q.mean()
    
    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min')
        
        for idx, prior_critic in enumerate(self._prior_critic_approximators):
            # # 'Next_Q' value should be a combination of prior task and current task residual_q (rho) values
            # Use weights for the prior rho values. Also use appropriate state-spaces as per the prior task
            weights, prior_state =  self._mdp_get_prior_state_fn(next_state, ik_task=(idx==0), reach_task=(idx==1)) # task[0] is an IK prior task
            rho_prior_next = weights * prior_critic.predict(prior_state, a, prediction='min')
            q += rho_prior_next
        
        if self._use_entropy:
            q -= self._alpha_np * log_prob_next

        q *= 1 - absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
