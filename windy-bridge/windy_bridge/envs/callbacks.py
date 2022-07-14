import os
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import time

from .algorithmic_baseline import get_optimal_step


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, mode, verbose=0, seed=42):
        super(CustomCallback, self).__init__(verbose)
        # general
        self.seed = seed
        self.mode = mode
        self.episodes = 50 # 50
        self.seed_list = [x * self.seed for x in range(0, (self.episodes))]
        print(self.seed_list)
        print(len(self.seed_list))
        self.eval_steps_per_episode = 400
        # metrics
        self.wins = 0
        self.avg_reward = []
        self.avg_difference = []
        self.steps = 0
        self.avg_commitment = []
        self.avg_steps_per_episode = []
        self.number_of_actions = []
        self.distance_traveled = []
        # metrics lists
        self.result_list_reward = []
        self.result_list_wins = []
        self.result_list_steps_per_win = []
        self.result_list_commitment = []
        self.result_list_steps_per_episode = []
        self.result_list_actions = []
        self.result_list_distance = []
        self.last_trajectories = []
        self.complete_distribution = []
        self.random_distribution_values = []
        self.result_list_difference = []
        # metrics config
        self.trajectory_number = 10
        self.last_distribution_values = []
        self.distribution_value_number = 10
        self.iterator = 0
        self.last_actions = []
        self.last_rewards = []
        # histogram logs
        self.histogram_agent_angles = []
        self.histogram_algo_angles = []
        self.histogram_distribution = []
        self.histogram_agent_y = []

        # Those variables will be accessible in the callback
        # (they are defined in the base class)

        # The RL model
        # self.model = None  # type: BaseAlgorithm

        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]

        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int

        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]

        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        print("-- training start --")
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.iterator += 1
        # only eval at every n-th rollout
        eval_frequency = 10
        if self.iterator % eval_frequency == 0:
            print("-- eval rollout reached --")
            print("-- iterator: {} at {}".format(self.iterator, str(datetime.now())))
            self.eval_at()
            self.result_list_wins.append(self.wins)
            self.result_list_reward.append(self.avg_reward)
            self.result_list_difference.append(self.avg_difference)
            self.result_list_steps_per_win.append(self.steps)
            self.result_list_commitment.append(self.avg_commitment)
            self.result_list_steps_per_episode.append(self.avg_steps_per_episode)
            self.result_list_actions.append(self.number_of_actions)
            self.result_list_distance.append(self.distance_traveled)
            self.wins, self.steps, self.avg_difference = 0, 0, 0

            # lineplot metrics
            self.avg_reward = []
            self.distance_traveled = []
            self.avg_commitment = []
            self.number_of_actions = []
            self.avg_steps_per_episode = []




            # write histogram data
            self.write_histograms(self.iterator)
            self.histogram_agent_angles, self.histogram_algo_angles, self.histogram_distribution, self.histogram_agent_y = [], [], [], []
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("-- training end --")
        self.write_results(rewards=self.result_list_reward, wins=self.result_list_wins,
                           steps_per_win=self.result_list_steps_per_win, commitment=self.result_list_commitment,
                           trajectory=self.last_trajectories, distribution=self.last_distribution_values,
                           avg_steps=self.result_list_steps_per_episode, complete_distribution=self.complete_distribution,
                           actions=self.result_list_actions, distance=self.result_list_distance,
                           random_distribution_values=self.random_distribution_values, baseline_difference=self.result_list_difference,
                           last_ten_actions=self.last_actions, last_ten_rewards=self.last_rewards)
        # todo all in one line
        # reset all logging results
        self.result_list_reward = []
        self.result_list_wins = []
        self.result_list_steps_per_win = []
        self.result_list_commitment = []
        self.result_list_steps_per_episode = []
        self.complete_distribution = []
        self.result_list_actions = []
        self.result_list_distance = []
        self.random_distribution_values = []
        self.result_list_difference = []
        pass

    def eval_at(self):
        '''
        This method serves as a test run after n rollouts.
        It will run for self.episodes episodes.
        Winning the environment is defined by the numeric value of self.eval_steps_per_episode
        '''
        env = self.model.get_env()
        model = self.model
        self.last_trajectories = [None] * self.trajectory_number
        self.last_distribution_values = [None] * self.distribution_value_number
        self.last_actions = [None] * self.trajectory_number
        self.last_rewards = [None] * self.trajectory_number
        # todo updated seed from seed list has to go here. otherwise each episode gets exact same seed. => wanted behavior: every X episodes get the same X seeds
        for i in range(0, self.episodes):
            env.seed(self.seed_list[i])
            # todo random distribution for action sampling
            # np.clip(np.random.normal(0, 33, 100), -100, 100)
            # np.random.uniform(-100, 100)
            # random_distri = np.clip(np.random.normal(0, 30, self.eval_steps_per_episode), -90, 90)
            # random_distri = np.random.uniform(-90, 90, self.eval_steps_per_episode)

            _actions = 0
            _commitment = 0
            _env_steps = 0
            self.avg_difference = 0

            # last ten logging
            episode_trajectory = []
            episode_distribution = []
            episode_actions = []
            episode_rewards = []

            dist_per_episode = []
            obs = env.reset()
            # One episode:
            for e in range(self.eval_steps_per_episode):
                action, _states = model.predict(obs)
                # todo overwriting action with random sample
                #action = [[int(random_distri[e])/100, 0.1]] # min
                #action = [[int(random_distri[e]) / 100, np.random.randint(10) / 10]] # dynamic
                #action = [[int(random_distri[e]) / 100, 1.0]] # max
                #action = [[0, 0.1]]

                optimal_angle = get_optimal_step(obs[0][0], obs[0][1])[0]
                obs, rewards, done, info = env.step(action)

                # logging
                _actions += 1
                # todo log random distribution value for plotting
                # self.random_distribution_values.append([random_distri[e]])
                _commitment += int(action[0][1]*10)
                _env_steps += _commitment
                self.avg_difference += abs(optimal_angle - action[0][0]*100)
                self.avg_reward.append(float(rewards))
                self.distance_traveled.append(float(info[0]["distance"]))

                # last ten logging
                episode_trajectory.append(list(obs[0]))
                episode_distribution.append(info[0]["wind_values"])
                episode_actions.append(action[0][0])
                episode_rewards.append(list(rewards)[0])

                # log histogram data
                self.histogram_agent_angles.append(action[0][0]*100)
                self.histogram_algo_angles.append(optimal_angle)
                self.histogram_distribution.append(info[0]["wind_values"][0])
                self.histogram_agent_y.append(obs[0][1])

                # check win / check done
                if e >= self.eval_steps_per_episode-1:
                    done = True
                    self.wins += 1
                    self.steps += _actions
                if done:
                    self.last_trajectories[i % self.trajectory_number] = episode_trajectory
                    self.last_distribution_values[i % self.distribution_value_number] = episode_distribution
                    self.last_actions[i % self.trajectory_number] = episode_actions
                    self.last_rewards[i % self.trajectory_number] = episode_rewards
                    break

            # logging
            for sublist in episode_distribution:
                for value in sublist:
                    dist_per_episode.append(value)
            self.complete_distribution.append(dist_per_episode)
            self.avg_steps_per_episode.append(_env_steps)
            self.number_of_actions.append(_actions)
            self.avg_commitment.append(_commitment / _actions)
            self.avg_difference = self.avg_difference / e

        # lineplot logging might break this
        try:
            self.steps = self.steps / self.wins
        except ZeroDivisionError:
            self.steps = self.eval_steps_per_episode
        # lineplot from self.avg_reward = self.avg_reward / self.episodes -> self.avg_reward = [self.avg_reward]
        # redundant code below
        self.avg_reward = self.avg_reward
        self.distance_traveled = self.distance_traveled
        self.avg_commitment = self.avg_commitment
        self.number_of_actions = self.number_of_actions
        self.avg_steps_per_episode = self.avg_steps_per_episode

        self.wins = self.wins / self.episodes
        self.avg_difference = self.avg_difference

    def write_results(self, rewards, wins, steps_per_win, commitment, trajectory, distribution, avg_steps,
                      complete_distribution, actions, distance, random_distribution_values, baseline_difference,
                      last_ten_actions, last_ten_rewards):
        '''
        Args:
            rewards: avg_reward per over episodes per evaluated rollout
            wins: win percentage per over episodes per evaluated rollout
            steps_per_win: how many steps had to be taken to achieve the goal
            commitment: avg commitment agent took (1 for min; 10 for max)
            trajectory: agents trajectories from last 10 episode after learning
            distribution: wind distribution from last 10 episode after learning
            avg_steps: avg number of steps agent took in an episode
            complete_distribution: all wind values through the whole evaluation process
            actions: avg number of actions the agent took in an episode
            distance: avg distance the agent covered in an episode
            random_distribution_values: only used when comparing to random action samples (WIP)
            baseline_difference: average absolute different in chosen angle between agent and algorithmic_baseline.py
        Returns:
            writes all metrics into different files
        '''

        if not os.path.exists("logs/{}/histograms".format(self.mode)):
            os.makedirs("logs/{}/histograms".format(self.mode))
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d-%H%M%S")
        with open("logs/{}/rewards_{}.txt".format(self.mode, dt_string), "w+") as a:
            a.write(str(rewards))
        with open("logs/{}/wins_{}.txt".format(self.mode, dt_string), "w+") as b:
            b.write(str(wins))
        with open("logs/{}/steps_per_win_{}.txt".format(self.mode, dt_string), "w+") as c:
            c.write(str(steps_per_win))
        with open("logs/{}/commitment_{}.txt".format(self.mode, dt_string), "w+") as d:
            d.write(str(commitment))
        with open("logs/{}/trajectory_{}.txt".format(self.mode, dt_string), "w+") as e:
            e.write(str(trajectory))
        with open("logs/{}/distribution_{}.txt".format(self.mode, dt_string), "w+") as f:
            f.write(str(distribution))
        with open("logs/{}/avg_steps_per_episode_{}.txt".format(self.mode, dt_string), "w+") as g:
            g.write(str(avg_steps))
        with open("logs/{}/complete_distribution_{}.txt".format(self.mode, dt_string), "w+") as h:
            h.write(str(complete_distribution))
        with open("logs/{}/number_of_actions_{}.txt".format(self.mode, dt_string), "w+") as i:
            i.write(str(actions))
        with open("logs/{}/distance_traveled_{}.txt".format(self.mode, dt_string), "w+") as j:
            j.write(str(distance))
        with open("logs/{}/random_distribution_values_{}.txt".format(self.mode, dt_string), "w+") as k:
            k.write(str(random_distribution_values))
        with open("logs/{}/optimum_deviation_{}.txt".format(self.mode, dt_string), "w+") as l:
            l.write(str(baseline_difference))
        with open("logs/{}/last_ten_actions_{}.txt".format(self.mode, dt_string), "w+") as m:
            m.write(str(last_ten_actions))
        with open("logs/{}/last_ten_rew{}.txt".format(self.mode, dt_string), "w+") as n:
            n.write(str(last_ten_rewards))


    def write_histograms(self, iterator):
        if not os.path.exists("logs/{}/histograms".format(self.mode)):
            os.makedirs("logs/{}/histograms".format(self.mode))
        with open("logs/{}/histograms/{}_agent_angles.txt".format(self.mode, iterator), "w+") as a:
            a.write(str(self.histogram_agent_angles))
        with open("logs/{}/histograms/{}_algo_angles.txt".format(self.mode, iterator), "w+") as b:
            b.write(str(self.histogram_algo_angles))
        with open("logs/{}/histograms/{}_distribution.txt".format(self.mode, iterator), "w+") as c:
            c.write(str(self.histogram_distribution))
        with open("logs/{}/histograms/{}_agent_y.txt".format(self.mode, iterator), "w+") as d:
            d.write(str(self.histogram_agent_y))

