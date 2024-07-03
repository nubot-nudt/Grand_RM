from tqdm import tqdm
import numpy as np
np.random.seed()

class Core(object):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None,
                 preprocessors=None, prior_pretrain_only=False, pretrain_sampling_batch_size=20,
                 use_data_prior=False, prior_eps=0.0):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of
                each fit;
            callback_step (Callback): callback to execute after each step;
            preprocessors (list): list of state preprocessors to be
                applied to state variables before feeding them to the
                agent.
            prior_pretrain_only (bool): tells us whether to only pretrain a policy with samples from a prior
            use_data_prior (bool): tells us whether to use a prior from mdp for biasing data collection

        """
        self.agent = agent
        self.mdp = mdp
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None
        self._preprocessors = preprocessors if preprocessors is not None else list()

        self._state = None

        self._prior_pretrain_only = prior_pretrain_only
        self._pretrain_sampling_batch_size = pretrain_sampling_batch_size
        self._use_data_prior = use_data_prior
        self._prior_eps = prior_eps
        self._prior_sample_count = 0
        self._prior_success_count = 0
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False, get_renders=False):
        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None)\
            or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition =\
                lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter\
                                     >= self._n_episodes_per_fit

        self._run(n_steps, n_episodes, fit_condition, render, quiet, get_renders, learning=True) # Add bool to signify this is a learning run

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False, get_renders=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            initial_states (np.ndarray, None): the starting states of each
                episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet, get_renders,
                         initial_states)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet, get_renders=False,
             initial_states=None, learning=False):
        assert n_episodes is not None and n_steps is None and initial_states is None\
            or n_episodes is None and n_steps is not None and initial_states is None\
            or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(
            initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            move_condition =\
                lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps,
                                      dynamic_ncols=True, disable=quiet,
                                      leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition =\
                lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(total=self._n_episodes,
                                         dynamic_ncols=True, disable=quiet,
                                         leave=False)

        return self._run_impl(move_condition, fit_condition, steps_progress_bar,
                              episodes_progress_bar, render, get_renders, initial_states, learning)

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  episodes_progress_bar, render, get_renders, initial_states, learning):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = list()
        self._prior_sample_count = 0
        self._prior_success_count = 0
        last = True        
        while move_condition():
            if last:
                if (learning and self._prior_pretrain_only): # in the pretrain prior learning case
                    # Only reset after a batch is complete
                    if (self._current_steps_counter%self._pretrain_sampling_batch_size == 0):
                        self.reset(initial_states)
                else:
                    self.reset(initial_states)

            sample = self._step(render, get_renders, learning)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)
            if fit_condition():
                self.agent.fit(dataset)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = list()

            last = sample[-1]

        self.agent.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset

    def _step(self, render, get_renders=False, learning=False):
        """
        Single step.

        Args:
            render (bool): whether to render or not.
            get_renders (bool): whether to return the render images
            learning (bool): tells us whether this is a learning step or an eval

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        # # # Modifications to use priors in learning:
        # if (learning & self._use_data_prior):
        #     if self.mdp.check_prior_condition():    # Data prior
        #         # Don't always draw action from policy, instead
        #         # bias action selection using prior (with epsilon probability. epsilon can be modified from the main script)            
        #         if (self._prior_eps >= np.random.uniform()):
        #             # use sample from prior
        #             action = self.mdp.get_prior_action() # don't need to pass _state. The mdp knows the state
        #             self._prior_sample_count += 1
        #             data_prior_used = True
        #         else:
        #             action = self.agent.draw_noisy_action(self._state) # draw noisy action for the behavior policy (+ gaussian noise)
        #             data_prior_used = False
        #         # Step environment (mdp)
        #         next_state, reward, absorbing, _ = self.mdp.step(action)
        #         if (data_prior_used and (reward >=0.5)):
        #             self._prior_success_count += 1
        if (learning and self._prior_pretrain_only): # Pretrain Prior
            self._episode_steps -= 1 # Don't count these samples as episode_steps
            if not(self.agent._replay_memory.initialized):
                # Generate samples from prior only (without stepping the environment) to
                # fill the replay buffer (until buffer is initialised)
                # Note that we also need to fill the buffer with other random samples, so use prior samples only with eps probability
                if (self._prior_eps >= np.random.uniform()):
                    action = self.mdp.get_prior_action() # don't need to pass _state. The mdp knows the state
                    next_state = np.zeros(self.mdp.info.observation_space.shape) # dummy value. Not relevant because we assume termination after getting max reward
                    reward = self.mdp._reward_success # TODO: + distance reward...
                    absorbing = True
                else:
                    action = np.hstack((np.random.uniform(size=3), np.array([0.,0.]))) # Action space for tiago reaching...
                    action[np.random.choice([3,4])] = 1.0 # Discrete action
                    next_state, reward, absorbing, _ = self.mdp.step(action) # TODO: Use ground truth values here instead of stepping
            else:
                # Relay buffer ready, don't step the mdp but just set dummy values
                action, next_state, reward, absorbing = np.zeros(self.mdp.info.action_space.shape), np.zeros(self.mdp.info.observation_space.shape), 0.0, False
                # Set agent flag to fit only and not add new data to replay buffer
                self.agent._freeze_data = True
        elif learning:
            action = self.agent.draw_noisy_action(self._state)
            next_state, reward, absorbing, info = self.mdp.step(action)
        else: # Default
            action = self.agent.draw_action(self._state)
            next_state, reward, absorbing, info = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = not(
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state
        if get_renders:
            img = self.mdp.get_render()
            return img, state, action, reward, next_state, absorbing, info, last
        return state, action, reward, next_state, absorbing, info, last

    def reset(self, initial_states=None):
        """
        Reset the state of the agent.

        """
        if initial_states is None\
            or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        self.agent.episode_start()
        self.agent.next_action = None
        self._episode_steps = 0

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self._preprocessors:
            state = p(state)

        return state
