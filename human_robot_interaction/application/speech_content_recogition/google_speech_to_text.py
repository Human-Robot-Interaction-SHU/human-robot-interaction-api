from google.cloud import speech

client = speech.SpeechClient()


async def transcribe_streaming(audio_generator):
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code='en-US'
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )

    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)

    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    for response in responses:
        for result in response.results:
            yield result.alternatives[0].transcript
