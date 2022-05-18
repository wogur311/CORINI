from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt)
def index(request):
    return render(request, "main/test.html")

@method_decorator(csrf_exempt)
def model_test_view(request):
    print(request.POST)
    return render(request, "main/model-test.html")

def intro_view(request):
    return render(request, "main/intro.html")
