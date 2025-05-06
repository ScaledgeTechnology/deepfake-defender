from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/progress/<str:task_type>/<str:task_id>/', consumers.AnalysisProgressConsumer.as_asgi()),
]