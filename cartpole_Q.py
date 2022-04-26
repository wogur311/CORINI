from datetime import datetime
from typing import Tuple

from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import time, math, random
import gym
import ray

#sub_project_name = CartPole-v1 #으로 이름 설정하고 돌리자.

@ray.remote
class ML_Q:
    def __init__(self, sub_project_name, episode_num, step=200):
        self.episode_num = episode_num
        self.step = step
        self.env = gym.make(sub_project_name)
        self.n_bins = (6, 12)
        self.lower_bounds = [self.env.observation_space.low[2], -math.radians(50)]
        self.upper_bounds = [self.env.observation_space.high[2], math.radians(50)]
        self.Q_table = np.zeros(self.n_bins + (self.env.action_space.n,))

    def discretizer(self, _, __, angle, pole_velocity) -> Tuple[int, ...]:
        """Convert continues state intro a discrete state"""
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))

    # print(Q_table.shape)

    def policy(self, state: tuple):
        """Choosing action based on epsilon-greedy policy"""
        return np.argmax(self.Q_table[state])

    def new_Q_value(self, reward: float, new_state: tuple, discount_factor=1) -> float:
        """Temperal diffrence for updating Q-value of state-action pair"""
        future_optimal_value = np.max(self.Q_table[new_state])  # 새로운 상태에서 가장 큰 리워드
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    # 새로운 상태에서 얻을 수 있는 가장 큰 리워드를 반환

    # Adaptive learning of Learning Rate
    def learning_rate(self, n: int, min_rate=0.01) -> float:
        """Decaying learning rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

    def exploration_rate(self, n: int, min_rate=0.1) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))

    def run(self):
        for e in range(self.episode_num):

            # Siscretize state into buckets
            current_state, done = self.discretizer(*self.env.reset()), False

            for i in range(self.step):

                # policy action
                action = self.policy(current_state)  # exploit

                if np.random.random() < self.exploration_rate(e):  # 설정한 탐험 확률에 해당될 경우, 탐험한다.
                    action = self.env.action_space.sample()  # explore

                # increment enviroment
                obs, reward, done, _ = self.env.step(action)  # env.step(action)을 통해 바뀐 상황(state)를 다시 obs, reward, done, _에 저장
                #print("obs-continuous :", obs)
                new_state = self.discretizer(*obs)  # 새로운 state를 이산변수로 바꾸어줌
                #print("obs-discrete :", new_state)

                # Update Q-Table
                lr = self.learning_rate(e)
                learnt_value = self.new_Q_value(reward, new_state)
                old_value = self.Q_table[current_state][action]
                self.Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value

                current_state = new_state

                # Render the cartpole environment
                # self.env.render()

                if done==True:
                    break

        # self.env.close()
        return 'done'

    def model_test(self, iter):
        for i in range(iter):
            current_state, done = self.discretizer(*self.env.reset()), False

            for j in range(self.step):
                action = self.policy(current_state)
                obs, reward, done, _ = self.env.step(action)
                current_state = self.discretizer(*obs)
                self.env.render()

                if done==True:
                    break

        self.env.close()

    def get_Q_table(self):
        return self.Q_table





                # def Q(episode_num, goal_step):
    #     # env = gym.make('CartPole-v1')
    #     # n_bins = (6, 12)
    #     # lower_bounds = [env.observation_space.low[2], -math.radians(50)]
    #     # upper_bounds = [env.observation_space.high[2], math.radians(50)]
    #
    #     def discretizer(_, __, angle, pole_velocity) -> Tuple[int, ...]:
    #         """Convert continues state intro a discrete state"""
    #         est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    #         est.fit([lower_bounds, upper_bounds])
    #         return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))
    #
    #     Q_table = np.zeros(n_bins + (env.action_space.n,))
    #     #print(Q_table.shape)
    #
    #     def policy(state: tuple):
    #         """Choosing action based on epsilon-greedy policy"""
    #         return np.argmax(Q_table[state])
    #
    #     def new_Q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    #         """Temperal diffrence for updating Q-value of state-action pair"""
    #         future_optimal_value = np.max(Q_table[new_state])  # 새로운 상태에서 가장 큰 리워드
    #         learned_value = reward + discount_factor * future_optimal_value
    #         return learned_value
    #     # 새로운 상태에서 얻을 수 있는 가장 큰 리워드를 반환
    #
    #     # Adaptive learning of Learning Rate
    #     def learning_rate(n: int, min_rate=0.01) -> float:
    #         """Decaying learning rate"""
    #         return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
    #
    #     def exploration_rate(n: int, min_rate=0.1) -> float:
    #         """Decaying exploration rate"""
    #         return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))
    #
    #     for e in range(episode_num):
    #
    #         # Siscretize state into buckets
    #         current_state, done = discretizer(*env.reset()), False
    #
    #         while done == False:
    #
    #             # policy action
    #             action = policy(current_state)  # exploit
    #
    #             if np.random.random() < exploration_rate(e):  # 설정한 탐험 확률에 해당될 경우, 탐험한다.
    #                 action = env.action_space.sample()  # explore
    #
    #             # increment enviroment
    #             obs, reward, done, _ = env.step(action)  # env.step(action)을 통해 바뀐 상황(state)를 다시 obs, reward, done, _에 저장
    #             print("obs-continuous :", obs)
    #             new_state = discretizer(*obs)  # 새로운 state를 이산변수로 바꾸어줌
    #             print("obs-discrete :", new_state)
    #
    #             # Update Q-Table
    #             lr = learning_rate(e)
    #             learnt_value = new_Q_value(reward, new_state)
    #             old_value = Q_table[current_state][action]
    #             Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value
    #
    #             current_state = new_state
    #
    #             # Render the cartpole environment
    #             #env.render()