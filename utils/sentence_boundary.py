"""
sentence_boundary.py
---------------------
Detects when a continuous-signing "utterance" has ended, so the buffered
frames can be sent to the recognizer + translator instead of waiting for
the user to press a key.

Heuristic (simple, robust, no extra model needed):
    1. Track whether hands are visible in each frame.
    2. Track frame-to-frame motion magnitude of the hand landmarks.
    3. If hands are absent OR motion stays below MOTION_EPSILON for
       PAUSE_FRAMES_THRESHOLD consecutive frames -> utterance boundary.
    4. A hard cap (MAX_UTTERANCE_FRAMES) forces a cut-off for very long,
       continuous signing so the system still feels "real-time".
"""

from collections import deque
import numpy as np
import config


class SentenceBoundaryDetector:
    def __init__(self):
        self.buffer = []                 # feature vectors for current utterance
        self._prev_feature = None
        self._still_frame_count = 0
        self._active = False
        self._total_motion = 0.0               # True once signing has started

    def reset(self):
        self.buffer = []
        self._prev_feature = None
        self._still_frame_count = 0
        self._active = False
        self._total_motion = 0.0 

    def update(self, feature_vec: np.ndarray, hands_present: bool):
        """Feed one frame's feature vector in. Returns:
            utterance (list[np.ndarray] or None): a completed utterance's
                buffered frames if a boundary was just detected, else None.
        """
        motion = 0.0
        if self._prev_feature is not None:
            motion = float(np.linalg.norm(feature_vec - self._prev_feature))
        self._prev_feature = feature_vec

        is_still = (not hands_present) or (motion < config.MOTION_EPSILON)

        if hands_present:
            self._active = True
            self.buffer.append(feature_vec)
            self._total_motion += motion 

        if is_still:
            self._still_frame_count += 1
        else:
            self._still_frame_count = 0

        # Boundary condition 1: sustained pause after some active signing
        if self._active and self._still_frame_count >= config.PAUSE_FRAMES_THRESHOLD:
            return self._flush()

        # Boundary condition 2: hard cap on utterance length
        if self._active and len(self.buffer) >= config.MAX_UTTERANCE_FRAMES:
            return self._flush()

        return None

    def _flush(self):
        utterance = self.buffer
        total_motion = self._total_motion   # capture BEFORE reset() wipes it
        self.reset()
        if len(utterance) < config.MIN_UTTERANCE_FRAMES:
            return None
        if total_motion < config.MIN_UTTERANCE_MOTION:
            return None                     # <-- add this: hand was still, not a real sign
        return utterance
