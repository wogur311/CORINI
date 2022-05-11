from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

def parsing(request):
    form_data = form = request.POST
    model1_data = {'model1_type': form_data['model1_type'],
                   'q_number_of_episode': int(form_data['q_number_of_episode']),
                   'q_learning_rate': float(form_data['q_learning_rate'])}

    model2_data = {'model2_type': form_data['model2_type'],
                   'ne_generation': int(form_data['ne_generation']),
                   'ne_population': int(form_data['ne_population']),
                   'ne_top_limit': int(form_data['ne_population'].replace("%", ""))}

    print(model1_data)
    print(model2_data)
    return 0

@method_decorator(csrf_exempt)
def index(request):
    return render(request, "main/test.html")
