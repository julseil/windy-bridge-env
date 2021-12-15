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
    def __init__(self, mode, verbose=0, seed=0):
        super(CustomCallback, self).__init__(verbose)
        # gerneral
        #self.seed = seed
        self.mode = mode
        self.episodes = 100
        self.eval_steps_per_episode = 500
        # metrics
        self.wins = 0
        self.losses = 0
        self.avg_reward = 0
        self.avg_difference = 0
        self.steps = 0
        self.avg_commitment = 0
        self.avg_steps_per_episode = 0
        self.number_of_actions = 0
        self.distance_traveled = 0
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
        #self.iterator += 1
        #if self.iterator % 50 == 0 or self.iterator == 1:
        #    print("%s : %s / 500" % (str(datetime.now()), self.iterator))
        self.eval_at()
        self.result_list_wins.append(self.wins)
        self.result_list_reward.append(self.avg_reward)
        self.result_list_difference.append(self.avg_difference)
        self.result_list_steps_per_win.append(self.steps)
        self.result_list_commitment.append(self.avg_commitment)
        self.result_list_steps_per_episode.append(self.avg_steps_per_episode)
        self.result_list_actions.append(self.number_of_actions)
        self.result_list_distance.append(self.distance_traveled)
        self.wins, self.losses, self.avg_reward, self.steps, self.avg_commitment, \
            self.avg_steps_per_episode, self.number_of_actions, self.distance_traveled,\
            self.avg_difference = 0, 0, 0, 0, 0, 0, 0, 0, 0
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.write_results(rewards=self.result_list_reward, wins=self.result_list_wins,
                           steps_per_win=self.result_list_steps_per_win, commitment=self.result_list_commitment,
                           trajectory=self.last_trajectories, distribution=self.last_distribution_values,
                           avg_steps=self.result_list_steps_per_episode, complete_distribution=self.complete_distribution,
                           actions=self.result_list_actions, distance=self.result_list_distance,
                           random_distribution_values=self.random_distribution_values, baseline_difference=self.result_list_difference)
        # todo all in one line
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
        env = self.model.get_env()
        model = self.model
        #env.seed(self.seed)
        self.last_trajectories = [None] * self.trajectory_number
        self.last_distribution_values = [None] * self.distribution_value_number
        for i in range(self.episodes):
            # todo random distribution for action sampling
            # np.clip(np.random.normal(0, 33, 100), -100, 100)
            # np.random.uniform(-100, 100)
            #random_distri = np.clip(np.random.normal(0, 30, self.eval_steps_per_episode), -90, 90)
            #random_distri = np.random.uniform(-90, 90, self.eval_steps_per_episode)
            _actions = 0
            _commitment = 0
            _env_steps = 0
            self.avg_difference = 0
            trajectory = []
            distribution = []
            dist_per_episode = []
            obs = env.reset()
            for e in range(self.eval_steps_per_episode):
                action, _states = model.predict(obs)
                # todo overwriting action with random sample
                #action = [[int(random_distri[e])/100, 0.1]]
                #action = [[0, 0.1]]

                optimal_angle = get_optimal_step(obs[0][0], obs[0][1])[0]
                obs, rewards, done, info = env.step(action)
                _actions += 1
                #self.random_distribution_values.append([random_distri[e]])
                _commitment += int(action[0][1]*10)
                _env_steps += _commitment
                self.avg_difference += abs(optimal_angle - action[0][0]*100)
                self.avg_reward += float(rewards)
                self.distance_traveled += float(info[0]["distance"])
                trajectory.append(list(obs[0]))
                distribution.append(list(info[0]["wind_values"]))
                #env.render()
                if e >= self.eval_steps_per_episode-1:
                    done = True
                    self.wins += 1
                    self.steps += _actions
                    print("Win")
                if done:
                    print("Done")
                    print(e)
                    self.last_trajectories[i % self.trajectory_number] = trajectory
                    self.last_distribution_values[i % self.distribution_value_number] = distribution
                    self.avg_difference = self.avg_difference / e
                    break
            for sublist in distribution:
                for value in sublist:
                    dist_per_episode.append(value)
            self.complete_distribution.append(dist_per_episode)
            self.avg_steps_per_episode = _env_steps
            self.number_of_actions = _actions
            self.avg_commitment += _commitment / _actions
            self.result_list_difference.append(self.avg_difference)

        try:
            self.steps = self.steps / self.wins
        except ZeroDivisionError:
            self.steps = self.eval_steps_per_episode
        self.wins = self.wins / self.episodes
        self.avg_reward = self.avg_reward / self.episodes
        self.avg_difference = self.avg_difference / self.episodes
        self.distance_traveled = self.distance_traveled / self.episodes
        self.avg_commitment = self.avg_commitment / self.episodes
        self.number_of_actions = self.number_of_actions / self.episodes
        self.avg_steps_per_episode = self.avg_steps_per_episode / self.episodes

    def write_results(self, rewards, wins, steps_per_win, commitment, trajectory, distribution, avg_steps,
                      complete_distribution, actions, distance, random_distribution_values, baseline_difference):
        # todo get mode for dir structure
        if not os.path.exists("logs/{}".format(self.mode)):
            os.makedirs("logs/{}".format(self.mode))
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


