from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from datetime import datetime
import time


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, seed, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.seed = seed
        self.test_runs = 200
        self.eval_steps_per_run = 1000
        self.wins = 0
        self.losses = 0
        self.avg_reward = 0
        self.steps = 0
        self.avg_commitment = 0
        self.result_list_reward = []
        self.result_list_wins = []
        self.result_list_steps_per_win = []
        self.result_list_commitment = []
        self.last_trajectories = []
        self.trajectory_number = 10
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
        self.result_list_steps_per_win.append(self.steps)
        self.result_list_commitment.append(self.avg_commitment)
        self.wins, self.losses, self.avg_reward, self.steps, self.avg_commitment = 0, 0, 0, 0, 0
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        #self.plot_results()
        self.write_results(rewards=self.result_list_reward, wins=self.result_list_wins,
                           steps_per_win=self.result_list_steps_per_win, commitment=self.result_list_commitment,
                           trajectory=self.last_trajectories)
        self.result_list_reward = []
        self.result_list_wins = []
        self.result_list_steps_per_win = []
        self.result_list_commitment = []
        pass

    def eval_at(self):
        env = self.model.get_env()
        model = self.model
        env.seed(self.seed)
        self.last_trajectories = [None] * self.trajectory_number
        for i in range(self.test_runs):
            _steps = 0
            _commitment = 0
            trajectory = []
            obs = env.reset()
            for e in range(self.eval_steps_per_run):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                _steps += 1
                _commitment += int(action[0][1]*10)
                self.avg_reward += float(rewards)
                trajectory.append(list(obs[0]))
                #env.render()
                if done:
                    self.last_trajectories[i % self.trajectory_number] = trajectory
                    if 9.8 < rewards < 10.1:
                        self.wins += 1
                        self.steps += _steps
                    else:
                        self.losses += 1
                    break

            self.avg_commitment += _commitment / _steps

        try:
            self.steps = self.steps / self.wins
        except ZeroDivisionError:
            self.steps = self.eval_steps_per_run
        self.wins = self.wins / self.test_runs
        self.avg_reward = self.avg_reward / self.test_runs
        self.avg_commitment = self.avg_commitment / self.test_runs

    def plot_results(self):
        y_reward = self.result_list_reward
        y_steps = self.result_list_steps_per_win
        y_wins = self.result_list_wins
        x = [i for i in range(len(y_wins))]
        plt.plot(x, y_wins)
        plt.show()

    def write_results(self, rewards, wins, steps_per_win, commitment, trajectory):
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d-%H%M%S")
        with open("logs/rewards_{}.txt".format(dt_string), "w") as a:
            a.write(str(rewards))
        with open("logs/wins_{}.txt".format(dt_string), "w") as b:
            b.write(str(wins))
        with open("logs/steps_per_win_{}.txt".format(dt_string), "w") as c:
            c.write(str(steps_per_win))
        with open("logs/commitment_{}.txt".format(dt_string), "w") as d:
            d.write(str(commitment))
        with open("logs/trajectory_{}.txt".format(dt_string), "w") as e:
            e.write(str(trajectory))
