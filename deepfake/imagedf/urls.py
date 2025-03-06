from django.urls import path

from . import views

urlpatterns = [
    path('',views.image_upload,name='image_upload'),
    path('display_image/', views.image_display, name='image_display'), 
   
]
