from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from cartpole_Q import *
from cartpole_NE_2 import *
import random

# model-select 에서 post로 받은 값들을 model-selected에 넣어준뒤
# model-test 에서는 각각의 모델에 맞는 설정값을 입력할 수 있다.
# 남은 것 : model-test view에서 받은 post 값으로 머신러닝 모델 구현
#       -> 모델 결과를 보여줄 result view를 생성해야함


# model-selecet에서 선택된 모델로 업데이트됨
model_selected = {'model1_type':'Q-learning',
                  'model2_type':'Neuroevolution'}

#  <QueryDict: {'q_number_of_episode': ['100'], 'q_learning_rate': ['0.7'],
#  'ne_generation': ['500'], 'ne_population': ['100'], 'ne_top_limit': ['5%'],
#  'model1_type': ['Q-learning'], 'model2_type': ['Neuroevolution']}> -> arg : model train 화면에서 입력받은 값들
args = {}

@method_decorator(csrf_exempt)
def index(request): # 1. 시작 화면.
    return render(request, "main/test.html")

@method_decorator(csrf_exempt)
def model_select_view(request): # 2. 모델 선택 화면
    global model_selected
    if request.POST:
        model_selected['model1_type'] = request.POST['model1_type']
        model_selected['model2_type'] = request.POST['model2_type']
        return redirect('model_test_view')
    return render(request, "main/model-select.html")

@method_decorator(csrf_exempt)
def model_test_view(request): # 3. 선택된 모델에 넣어줄 변수들을 설정하고 트레인 버튼을 누르는 화면
    global model_selected, args
    # 문제 : 같은 모델을 선택할 경우, 딕셔너리로 되어 있기 때문에 변수값 두 개가 모두 같이 저장됨. -> 일단 나중에 하자.
    if request.POST: # form 값을 post로 받으면, == train button pressed
        args['q_number_of_episode'] = request.POST['q_number_of_episode']
        args['q_learning_rate'] = request.POST['q_learning_rate']
        args['ne_generation'] = request.POST['ne_generation']
        args['ne_population'] = request.POST['ne_population']
        args['ne_top_limit'] = request.POST['ne_top_limit']
        args['model1_type'] = model_selected['model1_type']
        args['model2_type'] = model_selected['model2_type']

        return redirect('train_waiting_view')

    return render(request, "main/model-test.html", model_selected) # 그게 아니면 선택 화면을 보여줄 것.

def train_waiting_view(request):
    time = waiting_time() * 1000
    print(time)
    return render(request, "main/model-waiting.html", {'waiting_time' :time} )

def waiting_time():
    global args
    q_episode = int(args['q_number_of_episode'])
    ne_generation = int(args['ne_generation'])
    ne_population = int(args['ne_population'])
    ne_top_limit = (ne_population // 100) * int(args['ne_top_limit'][::len(args['ne_top_limit'])])

    waiting_time_q = q_time(q_episode)
    waiting_time_ne = ne_time(ne_generation, ne_population, ne_top_limit)

    args['waiting_time_q'] = waiting_time_q
    args['waiting_time_ne'] = waiting_time_ne

    time = max(waiting_time_q, waiting_time_ne)
    return time

def q_time(episode):
    if episode <= 100 :
        return 1 + random.randrange(0, 10) + random.random()

    if episode <= 500 :
        return 5 + random.randrange(0, 10) + random.random()

    if episode <= 1000 :
        return 70 + random.randrange(0, 10) + random.random()

    if episode <= 10000 :
        return 106 + random.randrange(0, 10) + random.random()
    else:
        return 200 + random.randrange(0, 10) + random.random()

def ne_time(generation, population, top_limit):
    if generation <= 50:
        time =  50 + random.randrange(0, 10) + random.random()
    elif generation > 50 and generation <= 100:
        time = 150
    elif generation > 100 and generation <= 500:
        time = 800
    elif generation > 500 and generation <= 1000:
        time = 1000
    else:
        time = 1500

    if population <= 100:
        time += 10 + random.randrange(0, 10)
    elif population > 100 and population <= 300:
        time += 30 + random.randrange(0, 10)
    elif population > 300 and population <= 500:
        time += 60 + random.randrange(0, 10)
    else:
        time += 100 + random.randrange(0, 10)

    if top_limit <= 5:
        time += random.random()
    elif top_limit > 5 and top_limit <= 20:
        time += 5 + random.random()
    elif top_limit > 20 and top_limit <= 30:
        time += 10 + random.random()
    else:
        time += 30 + random.random()

    return time

def model_result_view(request): # 모델 선택 후 여기에서 머신러닝 모델 생성, 훈련을 함. 훈련이 끝나면 결과를 render 해줄 것.
    global args

    ## model1 settings
    if args['model1_type'] == 'Q-learning':
        episode = int(args['q_number_of_episode'])
        learning_rate = float(args['q_learning_rate'])
        model1 = ML_Q(episode, learning_rate)
    else:
        generation = int(args['ne_generation'])
        population = int(args['ne_population'])
        top_limit = (population // 100) * int(args['ne_top_limit'][::len(args['ne_top_limit'])])
        model1 = ML_NE_2(generation, population, top_limit)

    ## model2 settings
    if args['model2_type'] == 'Q-learning':
        episode = int(args['q_number_of_episode'])
        learning_rate = float(args['q_learning_rate'])
        model2 = ML_Q(episode, learning_rate)

    else:
        generation = int(args['ne_generation'])
        population = int(args['ne_population'])
        top_limit = (population // 100) * int(args['ne_top_limit'][::len(args['ne_top_limit'])])
        model2 = ML_NE_2(generation, population, top_limit)

    return render(request, "main/model-result.html", args)


def intro_view(request): # 시작 화면에서 'about'을 누를 경우 넘어가는 화면. 우리 프로젝트, 팀 정보에 대해서 간략히 적어놓으면 좋을 듯함.
    return render(request, "main/intro.html")
