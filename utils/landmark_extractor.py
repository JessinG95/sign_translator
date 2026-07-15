"""
landmark_extractor.py
----------------------
Wraps MediaPipe Holistic to turn a single BGR camera frame into a flat,
fixed-length numpy feature vector containing:
    - pose landmarks (x, y, z, visibility) x 33
    - left hand landmarks (x, y, z) x 21
    - right hand landmarks (x, y, z) x 21
    - a sampled subset of face-mesh landmarks (x, y) -- for mouth/eyebrow
      shape, which carries grammatical information in many sign languages
      (e.g. negation, questions).

If a component (hand/face/pose) isn't detected in a frame, its slot is
filled with zeros so every frame always produces a vector of the same
length (config.FRAME_FEATURE_DIM). This keeps downstream sequence models
simple -- no ragged inputs.
"""

import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "mediapipe is required for landmark extraction. "
        "Install with: pip install mediapipe"
    ) from e

import config


class LandmarkExtractor:
    """Stateful wrapper around a single MediaPipe Holistic instance.

    Reuse ONE instance across frames (don't recreate per-frame) since
    model loading is expensive.
    """

    def __init__(self,
                 min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE):
        self._mp_holistic = mp.solutions.holistic
        self.holistic = self._mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )
        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles

    def close(self):
        self.holistic.close()

    def process(self, frame_bgr: np.ndarray):
        """Run MediaPipe on a single frame.

        Returns:
            feature_vec (np.ndarray): shape (config.FRAME_FEATURE_DIM,)
            results: raw MediaPipe results object (useful for drawing overlays)
            hands_present (bool): True if at least one hand was detected
                                   (used by sentence-boundary / pause logic)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)

        pose_vec = self._extract_pose(results)
        left_hand_vec = self._extract_hand(results.left_hand_landmarks)
        right_hand_vec = self._extract_hand(results.right_hand_landmarks)
        face_vec = self._extract_face_subset(results.face_landmarks)

        feature_vec = np.concatenate(
            [pose_vec, left_hand_vec, right_hand_vec, face_vec]
        ).astype(np.float32)

        hands_present = (results.left_hand_landmarks is not None or
                          results.right_hand_landmarks is not None)

        return feature_vec, results, hands_present

    # ------------------------------------------------------------------
    # Component extractors -- each returns a zero vector if not detected
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_pose(results) -> np.ndarray:
        if results.pose_landmarks is None:
            return np.zeros(config.POSE_FEATS, dtype=np.float32)
        pts = []
        for lm in results.pose_landmarks.landmark:
            pts.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _extract_hand(hand_landmarks) -> np.ndarray:
        if hand_landmarks is None:
            return np.zeros(config.NUM_HAND_LANDMARKS * 3, dtype=np.float32)
        pts = []
        for lm in hand_landmarks.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _extract_face_subset(face_landmarks) -> np.ndarray:
        if face_landmarks is None:
            return np.zeros(config.FACE_FEATS, dtype=np.float32)
        pts = []
        all_lms = face_landmarks.landmark
        for idx in config.FACE_LANDMARK_SUBSET:
            if idx < len(all_lms):
                lm = all_lms[idx]
                pts.extend([lm.x, lm.y])
            else:
                pts.extend([0.0, 0.0])
        return np.array(pts, dtype=np.float32)

    # ------------------------------------------------------------------
    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw pose/hand/face overlays on a frame for debugging / demo UI."""
        mp_holistic = self._mp_holistic
        if results.face_landmarks:
            self._drawing.draw_landmarks(
                frame_bgr, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._drawing_styles.get_default_face_mesh_contours_style(),
            )
        if results.pose_landmarks:
            self._drawing.draw_landmarks(
                frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            self._drawing.draw_landmarks(
                frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self._drawing.draw_landmarks(
                frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        return frame_bgr
