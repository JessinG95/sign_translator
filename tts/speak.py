"""
speak.py
---------
Text-to-speech wrapper. Runs speech synthesis in a background thread so
it never blocks the real-time video loop.

Two backends:
    - "pyttsx3": fully offline, low latency, good default for real-time use.
    - "gtts":     Google TTS, more natural voice but requires internet and
                  writes/plays an mp3 per sentence (higher latency).
"""

import queue
import threading
import tempfile
import os

import config


class Speaker:
    def __init__(self, engine: str = config.TTS_ENGINE,
                 rate: int = config.TTS_RATE, volume: float = config.TTS_VOLUME):
        self.engine_name = engine
        self.rate = rate
        self.volume = volume
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text: str):
        """Non-blocking: queues text to be spoken."""
        if text and text.strip():
            self._queue.put(text.strip())

    def stop(self):
        self._stop.set()
        self._queue.put(None)  # unblock the worker
        self._thread.join(timeout=2)

    # ------------------------------------------------------------------
    def _worker(self):
        if self.engine_name == "pyttsx3":
            self._worker_pyttsx3()
        elif self.engine_name == "gtts":
            self._worker_gtts()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_name}")

    def _worker_pyttsx3(self):
        import pyttsx3
        while not self._stop.is_set():
            text = self._queue.get()
            if text is None:
                break
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine

    def _worker_gtts(self):
        from gtts import gTTS
        try:
            from playsound import playsound
        except ImportError as e:
            raise ImportError(
                "The 'gtts' backend needs the 'playsound' package: pip install playsound"
            ) from e

        while not self._stop.is_set():
            text = self._queue.get()
            if text is None:
                break
            try:
                tts = gTTS(text=text, lang="en")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp_path = f.name
                tts.save(tmp_path)
                playsound(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)