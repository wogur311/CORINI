import time

from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# model-select 에서 post로 받은 값들을 model-selected에 넣어준뒤
# model-test 에서는 각각의 모델에 맞는 설정값을 입력할 수 있다.
# 남은 것 : model-test view에서 받은 post 값으로 머신러닝 모델 구현
#       -> 모델 결과를 보여줄 result view를 생성해야함


# model-selecet에서 선택된 모델로 업데이트됨
model_selected = {'model1_type':'Q-learning',
                  'model2_type': 'Neuroevolution'}

args = {}

def ml_test(args):
    time.sleep(10)
    return args

@method_decorator(csrf_exempt)
def index(request):
    return render(request, "main/test.html")

@method_decorator(csrf_exempt)
def model_select_view(request):
    global model_selected
    if request.POST:
        model_selected['model1_type'] = request.POST['model1_type']
        model_selected['model2_type'] = request.POST['model2_type']
        return redirect('model_test_view')
    return render(request, "main/model-select.html")

@method_decorator(csrf_exempt)
def model_test_view(request):
    global model_selected
    if request.POST:
        arg = request.POST.copy()
        arg['model1_type'] = model_selected['model1_type']
        arg['model2_type'] = model_selected['model2_type']
        print(arg)
        return render(request, 'main/model-result.html', arg)
    return render(request, "main/model-test.html", model_selected)

def model_result_view(request):
    global args
    print(model_selected)
    return render(request, "main/model-result.html", model_selected)

def intro_view(request):
    return render(request, "main/intro.html")
