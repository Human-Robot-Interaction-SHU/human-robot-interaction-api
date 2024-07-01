import os
import asyncio
import wave
import shutil
from datetime import datetime
import json

from transformers import (
    AutoTokenizer
)
from .speech_content_model import BertForMultiLabelClassification, MultiLabelPipeline

from .google_speech_to_text import transcribe_streaming

tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
pipeline = MultiLabelPipeline(model=model, tokenizer=tokenizer, threshold=0.3)


class ContentOfSpeechEmotionRecognizer:
    def __init__(self):
        self.audio_buffer = b''
        self.buffer_size_limit = 1024 * 1024  # 10 MB limit for buffer
        self.buffer_timeout = 20  # 20 seconds timeout for sending buffer
        self.session_id = None
        self.session_folder = None
        self.buffer_count = 0
        self.send_task = None
        self.session_audio_filename = None
        self.session_transcription_filename = None
        self.audio_chunks = []

        # Initialize tokenizer, model, and pipeline
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
        self.pipeline = MultiLabelPipeline(model=self.model, tokenizer=self.tokenizer, threshold=0.3)

    def initialize_session(self):
        # Generate a unique session ID (you can use a timestamp or any unique identifier)
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.session_folder = os.path.join("session_data", self.session_id)
        os.makedirs(self.session_folder, exist_ok=True)
        self.session_audio_filename = os.path.join(self.session_folder, "session_audio.wav")
        self.session_transcription_filename = os.path.join(self.session_folder, "session_transcription.txt")

        # Create or open session audio file in write binary mode if it doesn't exist
        if not os.path.exists(self.session_audio_filename):
            with wave.open(self.session_audio_filename, 'wb') as session_audio_wf:
                session_audio_wf.setnchannels(1)
                session_audio_wf.setsampwidth(2)  # 2 bytes (16 bits)
                session_audio_wf.setframerate(48000)  # 48 kHz

    async def receive_audio_data(self, bytes_data):
        self.audio_buffer += bytes_data
        self.audio_chunks.append(bytes_data)
        if len(self.audio_buffer) >= self.buffer_size_limit:
            return await self.process_buffer()
        else:
            if not self.send_task:
                self.send_task = asyncio.create_task(self.send_buffer_after_timeout())
        return None

    async def send_buffer_after_timeout(self):
        try:
            await asyncio.sleep(self.buffer_timeout)
            return await self.process_buffer()
        except asyncio.CancelledError as e:
            print(f"send_buffer_after_timeout task was cancelled: {e}")
            return None


    async def process_buffer(self, write_to_storage=False):
        if not self.audio_buffer:
            return None

        audio_content = self.audio_buffer
        self.audio_buffer = b''
        if self.send_task:
            self.send_task.cancel()
            self.send_task = None

        if write_to_storage:
            buffer_folder = os.path.join(self.session_folder, f"buffer_{self.buffer_count}")
            os.makedirs(buffer_folder, exist_ok=True)

            audio_filename = os.path.join(buffer_folder, "audio.wav")
            temp_filename = "temp_audio.wav"

            # Create or open a temporary file to append new data
            with wave.open(temp_filename, 'wb') as temp_wf:
                temp_wf.setnchannels(1)
                temp_wf.setsampwidth(2)  # 2 bytes (16 bits)
                temp_wf.setframerate(48000)  # 48 kHz

                # Write existing audio file content to temporary file
                if os.path.exists(audio_filename):
                    with wave.open(audio_filename, 'rb') as original_wf:
                        temp_wf.writeframes(original_wf.readframes(original_wf.getnframes()))

                # Append new audio data
                temp_wf.writeframes(audio_content)

            # Replace the original file with the temporary file
            shutil.move(temp_filename, audio_filename)

        audio_generator = (chunk for chunk in self.audio_chunks)
        text = " ".join([t async for t in transcribe_streaming(audio_generator)])

        if write_to_storage:
            transcription_filename = os.path.join(buffer_folder, "text.txt")
            with open(transcription_filename, 'w') as text_file:
                text_file.write(text)

        result = self.pipeline(text)
        for item in result:
            item['scores'] = [float(score) for score in item['scores']]
        json_response = {
            "transcription": text,
            "emotion_detection": result
        }
        json_result = json.dumps(json_response)

        self.buffer_count += 1
        self.audio_chunks = []

        if write_to_storage:
            # Append to session audio file in append binary mode
            with open(self.session_audio_filename, 'ab') as session_audio_file:
                session_audio_file.write(audio_content)

            # Append to session transcription file
            with open(self.session_transcription_filename, 'a') as session_transcription_file:
                session_transcription_file.write(text + "\n")

        return json_result
