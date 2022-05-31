from cartpole_Q import *
from cartpole_NE_2 import *

# Q-learning
Q_model = ML_Q(1000, 0.7) # 모델 생성
print(Q_model.is_done())
Q_model.model_train() # 훈련
print(Q_model.is_done())

print("시간",Q_model.get_train_time()) # 훈련하는 데 걸린 시간
print("q리워드",Q_model.get_train_rewards()) # 훈련한 에피소드 마다의 리워드

print('=========')

# NE
NE_model = ML_NE_2(50, 100, 20) # 모델 생성
NE_model.model_train() # 훈련

print("ne 시간",NE_model.get_train_time()) # 훈련하는 데 걸린 시간
print("ne리워드",NE_model.get_train_rewards()) # 훈련한 세대 마다 성능 좋은 agent의 리워드