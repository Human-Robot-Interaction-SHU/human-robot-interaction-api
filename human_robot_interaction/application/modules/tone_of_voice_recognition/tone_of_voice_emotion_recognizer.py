from transformers import pipeline
import torch
import librosa
import wave
import io
from pydub import AudioSegment


def create_wav(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="aac")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()


# def create_wav(frames):
#     output = io.BytesIO()
#     with wave.open(output, "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(44100)
#         wf.writeframes(b''.join(frames))
#     output.seek(0)  # Rewind the BytesIO object to the beginning
#     return output


def predict_emotion(audio_stream):
    # Check if MPS is available
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load the pretrained model
    classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                          device=device)

    # Load your audio file
    # use this if it is a stream being passed, and you want to convert to a wav file
    audio_file = create_wav(audio_stream)

    # Use this if it is a wav file directly being passed
    # audio_file = audio_stream

    # Ensure the audio_file is rewound to the beginning
    audio_file.seek(0)

    # Load audio using librosa
    audio, sr = librosa.load(audio_file, sr=16000)

    # Perform emotion classification
    result = classifier(audio)

    # Print the results
    for emotion in result:
        print(f"Emotion: {emotion['label']}, Score: {emotion['score']:.4f}")

    # Sort the results by score in descending order and get the top emotion
    main_emotion = sorted(result, key=lambda x: x['score'], reverse=True)[0]

    # Print only the top emotion
    print(f"Top Emotion: {main_emotion['label']}, Score: {main_emotion['score']:.4f}")

    return main_emotion['label']
