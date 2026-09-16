import base64
import os
import tempfile
import traceback

import requests
import runpod
import torch
from transformers import pipeline


MODEL_ID = os.getenv(
    "PARAKEET_MODEL",
    "nvidia/parakeet-tdt-0.6b-v3"
)

print("Loading Parakeet model:", MODEL_ID)

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=device
)

print("Parakeet loaded successfully")


def handler(job):
    temp_path = None

    try:
        data = job.get("input", {})

        audio_url = data.get("audio_url")
        audio_base64 = data.get("audio_base64")

        if not audio_url and not audio_base64:
            return {
                "error": "Provide audio_url or audio_base64"
            }

        if audio_url:
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            audio_bytes = response.content
        else:
            if audio_base64.startswith("data:"):
                audio_base64 = audio_base64.split(",", 1)[1]

            audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False
        ) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        result = asr(temp_path)

        text = result.get("text", "").strip()

        return {
            "transcription": text,
            "text": text,
            "model": MODEL_ID,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

    except Exception as exc:
        traceback.print_exc()

        return {
            "error": "Transcription failed",
            "detail": str(exc)
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


runpod.serverless.start({
    "handler": handler
})
