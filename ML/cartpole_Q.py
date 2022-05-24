from datetime import datetime
from typing import Tuple

from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import time, math, random
import gym
import ray

#@ray.remote
class ML_Q:
    def __init__(self, episode_num, learning_rate, cartpole_version='CartPole-v1', step=200):
        self.episode_num = episode_num # 사용자로부터 받은 episode 수
        self.learning_rate_custom = learning_rate # 사용자로부터 받은 laerning rate
        self.start_time = None # 첫번째 에피소드 시작 시간
        self.end_time = None # 마지막 에피소드 종료 시간

        self.rewards_of_episodes = []
        self.curr_episode = 0
        self.step = step
        self.env = gym.make(cartpole_version)
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

    def model_train(self):
        self.start_time = time.time()
        for curr_episode in range(self.episode_num):
            rewards = 0 # 매 에피소드마다 저장할 reward 총합
            self.curr_episode = curr_episode
            # Siscretize state into buckets
            current_state, done = self.discretizer(*self.env.reset()), False

            for i in range(self.step):

                # policy action
                action = self.policy(current_state)  # exploit

                if np.random.random() < self.exploration_rate(curr_episode):  # 설정한 탐험 확률에 해당될 경우, 탐험한다.
                    action = self.env.action_space.sample()  # explore

                # increment enviroment
                obs, reward, done, _ = self.env.step(action)  # env.step(action)을 통해 바뀐 상황(state)를 다시 obs, reward, done, _에 저장
                #print("obs-continuous :", obs)
                new_state = self.discretizer(*obs)  # 새로운 state를 이산변수로 바꾸어줌
                #print("obs-discrete :", new_state)

                # Update Q-Table
                lr = self.learning_rate(curr_episode)
                learnt_value = self.new_Q_value(reward, new_state)
                old_value = self.Q_table[current_state][action]
                self.Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value

                current_state = new_state
                rewards += reward
                # Render the cartpole environment
                # self.env.render()

                if done==True:
                    break

            self.rewards_of_episodes.append(rewards) # 매 에피소드마다 저장

            # self.env.close()
        self.end_time = time.time()
        return 'done'

    def play_agent(self, iter):
        for i in range(iter):
            r = 0
            current_state, done = self.discretizer(*self.env.reset()), False

            for j in range(self.step):
                action = self.policy(current_state)
                obs, reward, done, _ = self.env.step(action)
                current_state = self.discretizer(*obs)
                #self.env.render()
                r += reward
                if done==True:
                    break
            print('reward:', r, end=' ')

        self.env.close()

    def get_Q_table(self):
        return self.Q_table

    def get_train_time(self) -> str:
        """
        :return: whole training time (ms)
        """
        return str(self.end_time - self.start_time)

    def get_train_rewards(self) -> list:
        return self.rewards_of_episodes