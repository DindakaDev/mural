import webrtcvad


class TurnDetector:
    """Buffers 16kHz mono 16-bit PCM, classifies 20ms frames with WebRTC
    VAD, and reports a closed turn once >=500ms of trailing silence
    follows speech."""

    SAMPLE_RATE = 16000
    FRAME_MS = 20
    FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # int16 = 2 bytes/sample
    SILENCE_MS_TO_CLOSE = 500

    def __init__(self, vad_mode: int = 2):
        self._vad = webrtcvad.Vad(vad_mode)
        self._buffer = bytearray()
        self._turn_audio = bytearray()
        self._in_speech = False
        self._silence_ms = 0

    def feed(self, pcm_bytes: bytes) -> bytes | None:
        self._buffer += pcm_bytes
        closed_turn = None
        while len(self._buffer) >= self.FRAME_BYTES:
            frame = bytes(self._buffer[: self.FRAME_BYTES])
            del self._buffer[: self.FRAME_BYTES]
            is_speech = self._vad.is_speech(frame, self.SAMPLE_RATE)
            if is_speech:
                self._in_speech = True
                self._silence_ms = 0
                self._turn_audio += frame
            elif self._in_speech:
                self._turn_audio += frame
                self._silence_ms += self.FRAME_MS
                if self._silence_ms >= self.SILENCE_MS_TO_CLOSE:
                    closed_turn = bytes(self._turn_audio)
                    self._turn_audio = bytearray()
                    self._in_speech = False
                    self._silence_ms = 0
        return closed_turn
