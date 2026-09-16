import asyncio
import requests
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from kokoro_onnx.session import create_session
from kokoro_onnx.tokenizer import Tokenizer
from huggingface_hub import hf_hub_download
import soundfile as sf
import io
import numpy as np
import torch

app = FastAPI()

print("🚀 Cargando Whisper (STT)...")
stt_model = WhisperModel("small", device="cpu", compute_type="int8")

print("🚀 Cargando Kokoro (TTS)...")
onnx_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="onnx/model.onnx")

print("📥 Descargando voces...")
voices_files = {
    "af_bella": hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices/af_bella.bin"),
    "am_adam": hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices/am_adam.bin"),
    "ef_dora": hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/ef_dora.pt"),
    "em_alex": hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/em_alex.pt")
}

def prepare_voice(path):
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            # Extraer el tensor principal si viene dentro de un dict
            data = data.get("weight", next(iter(data.values())))
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        raw = data.astype(np.float32).flatten()
    else:
        raw = np.fromfile(path, dtype=np.float32)

    if raw.size >= 256:
        return raw[:256].reshape(1, 256)
    return np.atleast_2d(raw)

# 2. Instanciar Kokoro
session = create_session(onnx_path)

original_run = session.run
def safe_run(output_names, input_feed, run_options=None):
    fixed_feed = {}
    for k, v in input_feed.items():
        arr = np.array(v)
        if k == "style" or "style" in k:
            if arr.ndim == 1:
                arr = np.expand_dims(arr, axis=0)
            arr = arr[:, :256]
        elif arr.ndim == 1 and k != "speed":
            arr = np.expand_dims(arr, axis=0)
        fixed_feed[k] = arr
    return original_run(output_names, fixed_feed, run_options)

session.run = safe_run

tts_model = Kokoro.__new__(Kokoro)
tts_model.session = session
tts_model.sess = session
tts_model.has_timings = False
tts_model.espeak_config = None

# Cargar todas las voces
tts_model.voices = {name: prepare_voice(path) for name, path in voices_files.items()}
tts_model.tokenizer = Tokenizer()

def _get_voice_override(self, name, *args, **kwargs):
    return self.voices.get(name, list(self.voices.values())[0])

tts_model.get_voice = _get_voice_override.__get__(tts_model, Kokoro)
if hasattr(tts_model, "_get_voice"):
    tts_model._get_voice = _get_voice_override.__get__(tts_model, Kokoro)

ONNX_TO_NUMPY_DTYPE = {
    "tensor(int64)": np.dtype("int64"),
    "tensor(int32)": np.dtype("int32"),
    "tensor(float)": np.dtype("float32"),
    "tensor(float16)": np.dtype("float16"),
    "tensor(double)": np.dtype("float64"),
}

inputs = session.get_inputs()
tts_model._tokens_input = "tokens" if any(i.name == "tokens" for i in inputs) else inputs[0].name
tts_model._style_input = "style" if any(i.name == "style" for i in inputs) else inputs[1].name
tts_model._speed_input = "speed" if any(i.name == "speed" for i in inputs) else (inputs[2].name if len(inputs) > 2 else None)

tts_model._input_dtypes = {
    i.name: ONNX_TO_NUMPY_DTYPE.get(i.type, np.dtype("int64")) for i in inputs
}

PREFERRED_GENDER = "female"

print("🟢 Servidor listo.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Cliente conectado vía WebSocket")
    
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            audio_buffer = io.BytesIO(audio_bytes)
            
            # 1. Transcribir
            segments, info = stt_model.transcribe(audio_buffer)
            user_text = " ".join([s.text for s in segments]).strip()
            
            if not user_text or "amara.org" in user_text.lower():
                continue
                
            print(f"🎙️ Transcrito ({info.language}): {user_text}")

            # 2. Ollama LLM
            system_prompt = "Reply in the same language as the user input in 1 short sentence."
            res = requests.post("http://localhost:11434/api/generate", json={
                "model": "qwen2.5:7b",
                "prompt": f"{system_prompt}\nUser: {user_text}\nAssistant:",
                "stream": False
            }).json()
            reply_text = res.get("response", "").strip()
            print(f"🤖 LLM: {reply_text}")

            # 3. Mapear voz e idioma
            if info.language == "es":
                tts_lang = "es"
                selected_voice = "ef_dora" if PREFERRED_GENDER == "female" else "em_alex"
            else:
                tts_lang = "en-us"
                selected_voice = "af_bella" if PREFERRED_GENDER == "female" else "am_adam"

            print(f"🔊 Sintetizando con voz '{selected_voice}' ({tts_lang})...")
            samples, sample_rate = tts_model.create(reply_text, voice=selected_voice, speed=1.0, lang=tts_lang)
            
            out_buffer = io.BytesIO()
            sf.write(out_buffer, samples, sample_rate, format='WAV')
            
            # 4. Enviar respuesta
            await websocket.send_bytes(out_buffer.getvalue())

    except Exception as e:
        print(f"🔴 Cliente desconectado: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
