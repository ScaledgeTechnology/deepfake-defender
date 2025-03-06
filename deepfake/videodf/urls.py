from django.urls import path
from . import views

urlpatterns = [
    path('', views.video_upload, name='video_upload'),  
   path('trim_video/', views.trim_video, name='trim_video'),
   path('video_display/', views.video_display, name='video_display'),
    
]
