
class ML_Q:
    def __init__(self, episode_num, learning_rate, cartpole_version='CartPole-v1', step=200):
        self.episode_num = episode_num
        self.learning_rate = learning_rate
        self.result = []
        self.training_time = 50
        self.goal_reward_time = 30

    def train(self):
        if self.episode_num == 100:
            if self.learning_rate == 0.7:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

            elif self.learning_rate == 0.5:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

            else:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

        elif self.episode_num == 1000:
            if self.learning_rate == 0.7:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

            elif self.learning_rate == 0.5:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

            else:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

        elif self.episode_num == 10000:
            if self.learning_rate == 0.7:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

            elif self.learning_rate == 0.5:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

            else:
                self.training_time = 50
                self.goal_reward_time = 30
                self.result = []

class ML_NE:
    def __init__(self, generations, pop_size, top_limit, cartpole_version='CartPole-v1', max_step_per_epi=200):
        self.generations = generations
        self.pop_size = pop_size
        self.top_limit = top_limit

        self.result = []
        self.training_time = 100
        self.goal_reward_time = 20

    def train(self):
        if self.generations == 50:
            if self.pop_size == 100:
                pass
            elif self.pop_size == 300:
                pass
            else:
                pass

        elif self.generations == 100:
            if self.pop_size == 100:
                pass
            elif self.pop_size == 300:
                pass
            else:
                pass

        elif self.generations == 1000:
            if self.pop_size == 100:
                pass
            elif self.pop_size == 300:
                pass
            else:
                pass

        if self.top_limit == 5:
            self.training_time *= 1.2
        elif self.top_limit == 20:
            self.training_time *= 1.9
        elif self.top_limit == 30:
            self.training_time *= 2.2
