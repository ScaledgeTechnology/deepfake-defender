from django.shortcuts import render,redirect

def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def service(request):
    return render(request,'service.html')

def pricing(request):
    return render(request,'pricing.html')

def contact(request):
    return render(request,'contact.html')

def error_page(request):
    return render(request, "error.html")

