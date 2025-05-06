from django.urls import path
from . import views

urlpatterns = [
    path('', views.audio_upload, name='audio_upload'),  
    path('display_audio/', views.audio_display, name='audio_display'), 

    
]
