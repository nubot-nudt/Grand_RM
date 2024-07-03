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

from copy import deepcopy
from itertools import chain

# AWAC with a Hybrid action space (A sequential discrete approximator takes as input the continous action and outputs the discrete part of the action)

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


class AWAC_hybridPolicy(Policy):
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
            _temperature='torch',
            _gauss_noise_cov='numpy'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(state).detach().cpu().numpy()

    def draw_noisy_action(self, state):
        # Add clipped gaussian noise (only to the continuous actions!)
        cont_noise = np.random.multivariate_normal(np.zeros(self._mu_approximator.output_shape[0]),np.eye(self._mu_approximator.output_shape[0])*self._gauss_noise_cov)
        noise = np.hstack((cont_noise,np.zeros(self._discrete_approximator.output_shape[0])))
        return np.clip(self.compute_action_and_log_prob_t(state).detach().cpu().numpy() + noise, np.hstack((self._min_a,np.zeros(self._discrete_approximator.output_shape[0]))), np.hstack((self._max_a,np.ones(self._discrete_approximator.output_shape[0]))))

    def compute_action(self, state):
        """
        Function that samples actions using the reparametrization trick.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled as numpy arrays.

        """
        a = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=False):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, False): whether to compute the log
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


class AWAC_hybrid(DeepAC):
    """
    AWAC with a Hybrid action space (A sequential discrete approximator takes as input the 
    continous action and outputs the discrete part of the action)

    """
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_discrete_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 log_std_min=-20, log_std_max=2, temperature=1.0, gauss_noise_cov=0.0,
                 prior_agent_to_reuse=None, critic_fit_params=None):
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
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            temperature (float): the temperature for the softmax part of the gumbel reparametrization
            gauss_noise_cov ([float, Parameter]): Add gaussian noise to the drawn actions (if calling 'draw_noisy_action()')
            prior_agent_to_reuse: (Optional) prior agent to continue training with
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._freeze_data = False # Flag to fit critic and policy only and not use any new data
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)


        # if(prior_agent_to_reuse is not None):
        #     # Continue training with the prior agent's Q function
        #     self._critic_approximator = prior_agent_to_reuse[0]._critic_approximator
        #     self._target_critic_approximator = prior_agent_to_reuse[0]._target_critic_approximator
        # else:
        self._critic_approximator = Regressor(TorchApproximator,
                                          **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                 **target_critic_params)

        actor_mu_approximator = Regressor(TorchApproximator,
                                          **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator,
                                             **actor_sigma_params)
        actor_discrete_approximator = Regressor(TorchApproximator,
                                             **actor_discrete_params)
        self._actor_last_loss = None # Store actor loss for logging
        
        policy = AWAC_hybridPolicy(actor_mu_approximator,
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

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters(),
                                  actor_discrete_approximator.model.network.parameters())

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        if not(self._freeze_data): # flag to fit only and not use any new data
            self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            if self._replay_memory.size > self._warmup_transitions():
                action_new = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new)
                self._optimize_actor_parameters(loss)
                self._actor_last_loss = loss.detach().cpu().numpy() # Store actor loss for logging

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def _loss(self, state, action_new):
        q_0 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return  -q.mean()

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
        a = self.policy.compute_action(next_state)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min')
        q *= 1 - absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())
