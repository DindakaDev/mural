import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from kokoro_onnx.session import create_session
from kokoro_onnx.tokenizer import Tokenizer
from huggingface_hub import hf_hub_download
import torch

stt_model: WhisperModel | None = None
tts_model: Kokoro | None = None


def load_models() -> None:
    global stt_model, tts_model
    if stt_model is not None and tts_model is not None:
        return

    print("Cargando Whisper (STT)...")
    stt_model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Cargando Kokoro (TTS)...")
    onnx_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="onnx/model.onnx")

    print("Descargando voces...")
    voices_files = {
        "af_bella": hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices/af_bella.bin"),
        "am_adam": hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices/am_adam.bin"),
        "ef_dora": hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/ef_dora.pt"),
        "em_alex": hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/em_alex.pt"),
    }

    def prepare_voice(path):
        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                data = data.get("weight", next(iter(data.values())))
            if isinstance(data, torch.Tensor):
                # NOTE: torch's numpy() bridge requires an ABI-compatible numpy
                # (torch built against numpy 1.x's C API). kokoro-onnx requires
                # numpy>=2.0.2, which breaks that bridge at runtime
                # ("RuntimeError: Numpy is not available"). .tolist() goes
                # through pure Python instead of the broken C-API bridge.
                data = np.array(data.tolist(), dtype=np.float32)
            raw = data.astype(np.float32).flatten()
        else:
            raw = np.fromfile(path, dtype=np.float32)
        if raw.size >= 256:
            return raw[:256].reshape(1, 256)
        return np.atleast_2d(raw)

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

    model = Kokoro.__new__(Kokoro)
    model.session = session
    model.sess = session
    model.has_timings = False
    model.espeak_config = None
    model.voices = {name: prepare_voice(path) for name, path in voices_files.items()}
    model.tokenizer = Tokenizer()

    def _get_voice_override(self, name, *args, **kwargs):
        return self.voices.get(name, list(self.voices.values())[0])

    model.get_voice = _get_voice_override.__get__(model, Kokoro)
    if hasattr(model, "_get_voice"):
        model._get_voice = _get_voice_override.__get__(model, Kokoro)

    ONNX_TO_NUMPY_DTYPE = {
        "tensor(int64)": np.dtype("int64"),
        "tensor(int32)": np.dtype("int32"),
        "tensor(float)": np.dtype("float32"),
        "tensor(float16)": np.dtype("float16"),
        "tensor(double)": np.dtype("float64"),
    }
    inputs = session.get_inputs()
    model._tokens_input = "tokens" if any(i.name == "tokens" for i in inputs) else inputs[0].name
    model._style_input = "style" if any(i.name == "style" for i in inputs) else inputs[1].name
    model._speed_input = "speed" if any(i.name == "speed" for i in inputs) else (inputs[2].name if len(inputs) > 2 else None)
    model._input_dtypes = {i.name: ONNX_TO_NUMPY_DTYPE.get(i.type, np.dtype("int64")) for i in inputs}

    tts_model = model
    print("Modelos listos.")


def transcribe(audio: np.ndarray) -> str:
    """audio: float32 PCM in [-1, 1], 16kHz mono."""
    assert stt_model is not None, "load_models() must run before transcribe()"
    segments, _info = stt_model.transcribe(audio)
    return " ".join(segment.text for segment in segments).strip()


def synthesize(text: str, voice: str, lang: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
    assert tts_model is not None, "load_models() must run before synthesize()"
    samples, sample_rate = tts_model.create(text, voice=voice, speed=speed, lang=lang)
    return samples, sample_rate
