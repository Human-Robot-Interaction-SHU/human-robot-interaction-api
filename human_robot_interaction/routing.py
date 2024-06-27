from django.urls import re_path
from .api.consumers import speech_content_consumer

websocket_urlpatterns = [
    re_path(r'ws/emotion_detection/$', speech_content_consumer.EmotionDetectionConsumer.as_asgi()),
]
