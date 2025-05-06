"""
ASGI config for deepfake project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

# import os
# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake.settings')

# application = get_asgi_application()



# -------------------
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
# import audiodf.routing
from common import routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            # audiodf.routing.websocket_urlpatterns
            routing.websocket_urlpatterns
        )
    ),
})
# -------------------