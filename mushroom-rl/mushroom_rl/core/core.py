from tqdm import tqdm
import numpy as np
from mushroom_rl.core.logger.console_logger import ConsoleLogger

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
            agent (Agent): the agent moving according to a policy;策略
            mdp (Environment): the environment in which the agent moves;环境，mdp即为train_task.py中的envs
            callbacks_fit (list): list of callbacks to execute at the end of
                each fit;在每次适应（fit）结束时执行的回调函数列表
            callback_step (Callback): callback to execute after each step;在每一步之后执行的回调函数
            preprocessors (list): list of state preprocessors to be
                applied to state variables before feeding them to the
                agent.状态预处理器列表，用于在将状态变量提供给代理之前对其进行预处理
            prior_pretrain_only (bool): tells us whether to only pretrain a policy with samples from a prior
            是否仅使用来自先前数据的样本对策略进行预训练
            use_data_prior (bool): tells us whether to use a prior from mdp for biasing data collection
            告诉我们是否使用环境提供的先验信息来偏置数据收集
        """
        # 代理对象，即执行动作的实体，根据策略在环境中移动
        self.agent = agent
        # 环境对象，代表代理所处的环境，定义了代理可以采取的动作和观察到的状态
        self.mdp = mdp
        # 在每次适应（fit）结束时执行的回调函数列表
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        # 在每一步之后执行的回调函数
        self.callback_step = callback_step if callback_step is not None else lambda x: None
        # 状态预处理器列表，用于在将状态变量提供给代理之前对其进行预处理
        self._preprocessors = preprocessors if preprocessors is not None else list()

        self._state = None

        # prior_pretrain_only (bool)：指示是否仅使用来自先前数据的样本对策略进行预训练
        self._prior_pretrain_only = prior_pretrain_only
        # 用于预训练的采样批次大小
        self._pretrain_sampling_batch_size = pretrain_sampling_batch_size
        # 指示是否使用环境提供的先验信息来偏置数据收集
        self._use_data_prior = use_data_prior
        # 先验信息的调节参数
        self._prior_eps = prior_eps
        # 用于记录来自先前数据的采样次数和成功采样次数的计数器
        self._prior_sample_count = 0
        self._prior_success_count = 0
        # 用于跟踪总的和当前的训练周期数和步数的计数器
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        # 用于设置训练过程中的一些参数和计数器
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

        # 用于调试打印备份信息
        self.console_logger = ConsoleLogger(log_name='')

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False, get_renders=False):
        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;要移动智能体的步数
            n_episodes (int, None): number of episodes to move the agent;要移动智能体的回合数
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;在每次策略适应（fit）之间移动的步数
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;在每次策略适应之间移动的回合数
            render (bool, False): whether to render the environment or not;是否渲染环境
            quiet (bool, False): whether to show the progress bar or not.是否显示进度条

        """
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None) \
               or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        # fit_condition返回一个布尔值，指示是否满足了进行策略适应的条件
        if n_steps_per_fit is not None:
            fit_condition = \
                lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter \
                                    >= self._n_episodes_per_fit

        # self.console_logger.info("n_steps_per_fit=%d, fit_condition=%s"%(n_steps_per_fit, fit_condition))

        self._run(n_steps, n_episodes, fit_condition, render, quiet, get_renders,
                  learning=True)  # Add bool to signify this is a learning run

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False, get_renders=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.
        作用是使用代理的策略在环境中移动代理，并进行评估
        Args:
            initial_states (np.ndarray, None): the starting states of each
                episode;表示每个回合的起始状态
            n_steps (int, None): number of steps to move the agent;表示要移动代理的步数
            n_episodes (int, None): number of episodes to move the agent;表示要移动代理的回合数
            render (bool, False): whether to render the environment or not;表示是否渲染环境
            quiet (bool, False): whether to show the progress bar or not.表示是否显示进度条

        """
        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet, get_renders,
                         initial_states)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet, get_renders=False,
             initial_states=None, learning=False):
        """

        Args:
            n_steps: 表示要移动代理的步数
            n_episodes:表示要移动代理的回合数
            fit_condition:用于判断是否满足策略适应的条件
            render:指示是否在移动代理时渲染环境
            quiet:指示是否在控制台中显示进度条
            get_renders:指示是否返回渲染结果
            initial_states:表示每个回合的初始状态
            learning:指示是否在学习过程中调用此方法

        Returns:

        """
        assert n_episodes is not None and n_steps is None and initial_states is None \
               or n_episodes is None and n_steps is not None and initial_states is None \
               or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(
            initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            move_condition = \
                lambda: self._total_steps_counter < n_steps
            # 显示进度条
            steps_progress_bar = tqdm(total=n_steps,
                                      dynamic_ncols=True, disable=quiet,
                                      leave=False)
            # 轮数进度条
            episodes_progress_bar = tqdm(disable=True)
        else:
            # 这部分用于显示进度条
            move_condition = \
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
                if learning and self._prior_pretrain_only:  # in the pretrain prior learning case
                    # Only reset after a batch is complete
                    if (self._current_steps_counter % self._pretrain_sampling_batch_size == 0):
                        self.reset(initial_states)
                else:
                    self.reset(initial_states)


            # 这里返回的其实是state
            sample = self._step(render, get_renders, learning)

            """绘制Q值函数部分在step中才会更新"""
            # 在这一步中测试，这里的self.mdp._task.states其实对应的是sample[3]即next_state
            # 测试1：获取所有的可能状态，并采用actor网络预测对应动作是否可行的方法，效果很不好
            # a_reach, log_prob_next_reach = self.agent.policy.compute_action_and_log_prob(self.mdp._task.states)
            #
            # q_reach = self.agent._target_critic_approximator.predict(
            #     self.mdp._task.states, a_reach, prediction='min')

            # 测试2：获取所有的可能状态，并直接采用可能的状态对应动作的方法
            # q_reach = self.agent._target_critic_approximator.predict(
            #     self.mdp._task.states, self.mdp._task.round_loc, prediction='min')

            # 测试3：获取所有的可能状态，并直接采用可能的状态对应动作的方法，极坐标转换后一一对应
            # q_reach = self.agent._target_critic_approximator.predict(
            #     self.mdp._task.states, self.mdp._task.next_action, prediction='min')

            # 测试4：以机器人当前状态为state，可能的下一个动作为action
            # q_reach = self.agent._target_critic_approximator.predict(
            #     self.mdp._task.robot_state, self.mdp._task.next_action, prediction='min')

            # 测试5：以机器人当前状态为state，当前action的后两维为actor网络的输出
            # a_reach, log_prob_next_reach = self.agent.policy.compute_action_and_log_prob(self.mdp._task.robot_state.numpy())
            # a_reach[:, :3] = self.mdp._task.next_action[:, :3]
            # q_reach = self.agent._target_critic_approximator.predict(
            #     self.mdp._task.robot_state, a_reach, prediction='min')

            # self.console_logger.info("当前机器人状态为%s，下一个动作为%s"%(self.mdp._task.robot_state, self.mdp._task.next_action))

            # 测试6：以机器人当前状态为state，以预测的动作为输出
            # a_reach, log_prob_next_reach = self.agent.policy.compute_action_and_log_prob(self.mdp._task.states)
            #
            # q_reach = self.agent._target_critic_approximator.predict(
            #     self.mdp._task.robot_state, a_reach, prediction='min')

            # 测试7：以机器人当前状态为state，不让robot运动
            # self.mdp._task.next_action[:, :3] = np.random.rand(3240, 3) * 0.01
            # q_reach = self.agent._target_critic_approximator.predict(
            #     self.mdp._task.states, self.mdp._task.next_action, prediction='min')
            # #
            # # # 显示在仿真环境中的部分
            # self.mdp._task.show_reachability(q_reach)
            # self.mdp.refresh()
            # input("机器人运动步数")
            """绘制Q值函数部分"""

            # self.console_logger.info("在core函数中的sample=%s" % (str(sample[3])))
            # self.console_logger.info("在core函数中的states=%s" % (str(self.mdp._task.states)))
            # self.console_logger.info("q_reach=%s"%(str(q_reach)))
            # input("测试")

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            # 这里的sample其实是_step函数返回的state, action, reward, next_state, absorbing, info, last的tuple类型
            # dataset一直append直到数据量达到initial_replay_size
            dataset.append(sample)

            # self.console_logger.info("sample=%s"%(str(sample)))

            # 先收集1024个数据，然后调用fit函数
            if fit_condition():
                # self.console_logger.info("适应策略")
                self.agent.fit(dataset)  # 训练
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
        # 当前是否处于学习模式，并且是否仅用于先验预训练
        if learning and self._prior_pretrain_only:  # Pretrain Prior
            # self.console_logger.info("调用learning=%s和self._prior_pretrain_only部分" % (learning))
            # 当前的 _episode_steps 减去 1
            self._episode_steps -= 1  # Don't count these samples as episode_steps
            # 如果代理的回放缓冲区尚未初始化，则生成来自先验的样本来填充回放缓冲区，直到缓冲区被初始化
            if not self.agent._replay_memory.initialized:
                # self.console_logger.info("not self.agent._replay_memory.initialized")
                # Generate samples from prior only (without stepping the environment) to
                # fill the replay buffer (until buffer is initialised)
                # Note that we also need to fill the buffer with other random samples, so use prior samples only with eps probability
                # 如果使用的先验采样概率 self._prior_eps 大于或等于一个随机生成的值，则从先验中获取行动，否则将生成一个随机动作
                if self._prior_eps >= np.random.uniform():
                    # self.console_logger.info("stream 2_2")
                    action = self.mdp.get_prior_action()  # don't need to pass _state. The mdp knows the state
                    next_state = np.zeros(
                        self.mdp.info.observation_space.shape)  # dummy value. Not relevant because we assume termination after getting max reward
                    reward = self.mdp._reward_success  # TODO: + distance reward...
                    absorbing = True
                else:  # 否则生成一个随机动作
                    # self.console_logger.info("stream 2_3")
                    action = np.hstack(
                        (np.random.uniform(size=3), np.array([0., 0.])))  # Action space for tiago reaching...
                    action[np.random.choice([3, 4])] = 1.0  # Discrete action
                    next_state, reward, absorbing, _ = self.mdp.step(
                        action)  # TODO: Use ground truth values here instead of stepping
            else:  # 下一个状态被设为全零，奖励设为最大奖励值（假设在获取最大奖励之后立即终止），并将 absorbing 标志设置为 True
                # Relay buffer ready, don't step the mdp but just set dummy values
                # self.console_logger.info("stream 1_2")
                action, next_state, reward, absorbing = np.zeros(self.mdp.info.action_space.shape), np.zeros(
                    self.mdp.info.observation_space.shape), 0.0, False
                # Set agent flag to fit only and not add new data to replay buffer
                self.agent._freeze_data = True
        elif learning:
            # self.console_logger.info("调用learning=%s部分"%(learning))
            action = self.agent.draw_noisy_action(self._state)
            next_state, reward, absorbing, info = self.mdp.step(action)
        else:  # Default，其实默认调用的就是Default部分
            # self.console_logger.info("调用Default部分")
            action = self.agent.draw_action(self._state)
            next_state, reward, absorbing, info = self.mdp.step(action)

        # self.console_logger.info(
        #     "Debug_rlmmbp:learning=%s, self._prior_pretrain_only=%s" % (str(learning), str(self._prior_pretrain_only)))


        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = not (
                self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state
        if get_renders:
            img = self.mdp.get_render()
            return img, state, action, reward, next_state, absorbing, info, last

        """这是一段调试代码"""
        # self.console_logger.info("在core函数中：state=%s"%(state))
        # self.console_logger.info("action=%s" % (action))
        # self.console_logger.info("reward=%s" % (reward))
        # self.console_logger.info("next_state=%s" % (next_state))
        # self.console_logger.info("absorbing=%s" % (absorbing))
        # self.console_logger.info("info=%s" % (info))
        # self.console_logger.info("last=%s" % (last))
        """这是一段调试代码"""

        # absorbing表示是否完成
        return state, action, reward, next_state, absorbing, info, last

    def reset(self, initial_states=None):
        """
        Reset the state of the agent.

        """
        if initial_states is None \
                or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        # print("self.mdp.reset(initial_state)=")
        # input(self.mdp.reset(initial_state))
        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        self.agent.episode_start()
        self.agent.next_action = None
        self._episode_steps = 0

        """绘制Q值函数部分"""
        # 在这一步中测试，这里的self.mdp._task.states其实对应的是sample[3]即next_state
        # 测试1：获取所有的可能状态，并采用actor网络预测对应动作是否可行的方法
        # a_reach, log_prob_next_reach = self.agent.policy.compute_action_and_log_prob(self.mdp._task.states)
        #
        # q_reach = self.agent._target_critic_approximator.predict(
        #     self.mdp._task.states, a_reach, prediction='min')

        # 测试2：获取所有的可能状态，并直接采用可能的状态对应动作的方法，极坐标转换前未一一对应
        # q_reach = self.agent._target_critic_approximator.predict(
        #     self.mdp._task.states, self.mdp._task.round_loc, prediction='min')

        # 测试3：获取所有的可能状态，并直接采用可能的状态对应动作的方法，极坐标转换后一一对应
        # q_reach = self.agent._target_critic_approximator.predict(
        #     self.mdp._task.states, self.mdp._task.next_action, prediction='min')

        # 测试4：以机器人当前状态为state，可能的下一个动作为action
        # q_reach = self.agent._target_critic_approximator.predict(
        #     self.mdp._task.robot_state, self.mdp._task.next_action, prediction='min')

        # 测试5：以机器人当前状态为state，当前action的后两维为actor网络的输出
        # input("测试类型,type(self.mdp._task.robot_state)=%s, self.mdp._task.states.shape=%s" % (
        # type(self.mdp._task.robot_state), str(self.mdp._task.robot_state.shape)))

        # 测试6：以机器人当前状态为state，以预测的动作为输出
        # a_reach, log_prob_next_reach = self.agent.policy.compute_action_and_log_prob(self.mdp._task.states)
        #
        # q_reach = self.agent._target_critic_approximator.predict(
        #     self.mdp._task.robot_state, a_reach, prediction='min')

        # 测试7：以机器人当前状态为state，不让robot运动
        # self.mdp._task.next_action[:, :3] = np.random.rand(3240, 3) * 0.01
        # q_reach = self.agent._target_critic_approximator.predict(
        #     self.mdp._task.states, self.mdp._task.next_action, prediction='min')
        #
        # # # # a_reach, log_prob_next_reach = self.agent.policy.compute_action_and_log_prob(self.mdp._task.robot_state.numpy())
        # # # # a_reach[:, :3] = self.mdp._task.next_action[:, :3]
        # # # # q_reach = self.agent._target_critic_approximator.predict(
        # # # #     self.mdp._task.robot_state, a_reach, prediction='min')
        # # #
        # self.mdp._task.show_reachability(q_reach)
        # self.mdp.refresh()
        # # input("初始化机器人运动")
        """绘制Q值函数部分"""

        """绘制桌子周围的Q函数部分"""
        # q_reach = self.agent._target_critic_approximator.predict(
        #     self.mdp._task.surrounding_area_states, self.mdp._task.surrounding_actions, prediction='min')
        #
        # self.mdp._task.show_surrounding_reachablity(q_reach)
        # self.mdp.refresh()
        # input("初始化过程")
        """绘制桌子周围的Q函数部分"""

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
