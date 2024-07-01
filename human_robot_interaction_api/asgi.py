"""
ASGI config for human_robot_interaction_api project.

It exposes the ASGI callable as a modules-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
import human_robot_interaction_api.routing

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'human_robot_interaction_api.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            human_robot_interaction_api.routing.websocket_urlpatterns
        )
    ),
})


#application = get_asgi_application()
