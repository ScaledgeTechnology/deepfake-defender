"""
URL configuration for deepfake project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name="index"),
    path('about/',views.about,name="about"),
    path('service/',views.service,name="service"),
    path('pricing/',views.pricing,name="pricing"),
    path('contact/',views.contact,name="contact"),
 
    path('videodf/', include('videodf.urls')), 
    path('audiodf/', include('audiodf.urls')), 
    path('imagedf/', include('imagedf.urls')), 

    path("error/", views.error_page, name="error_page"),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 

# ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) \
#   + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
