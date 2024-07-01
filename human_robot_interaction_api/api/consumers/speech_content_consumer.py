from channels.generic.websocket import AsyncWebsocketConsumer

from human_robot_interaction_api.application.modules.speech_content_recognition.content_of_speech_emotion_recognizer import \
    ContentOfSpeechEmotionRecognizer


class EmotionDetectionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recognizer = ContentOfSpeechEmotionRecognizer()

    async def connect(self):
        await self.accept()
        self.recognizer.initialize_session()

    async def disconnect(self, close_code):
        if self.recognizer.send_task:
            self.recognizer.send_task.cancel()

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            json_result = await self.recognizer.receive_audio_data(bytes_data)
            if json_result:
                await self.send(text_data=json_result)