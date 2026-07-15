"""
main.py
--------
Real-time Sign Language -> Voice Translator.

Pipeline per frame:
    webcam frame
        -> LandmarkExtractor (MediaPipe: hands + pose + face)
        -> SentenceBoundaryDetector (buffers frames, detects pauses)
    on utterance boundary:
        -> sliding-window CNN+BiLSTM SignRecognizer -> sequence of glosses
        -> Gloss2SentenceTransformer -> natural English sentence
        -> Speaker (TTS) -> spoken aloud, non-blocking

Run:
    python main.py

Requires trained checkpoints (see README.md for the training pipeline):
    checkpoints/sign_recognizer.pt
    checkpoints/gloss2sentence.pt
    checkpoints/gloss_vocab.json
    checkpoints/gloss_seq_vocab.json
    checkpoints/sentence_vocab.json
"""

import json
import time

import cv2
import numpy as np
import torch

import config
from utils.landmark_extractor import LandmarkExtractor
from utils.sentence_boundary import SentenceBoundaryDetector
from utils.vocab import Vocab, detokenize
from models.sign_recognizer import SignRecognizer
from models.seq2seq_transformer import Gloss2SentenceTransformer
from tts.speak import Speaker


class SignToVoiceTranslator:
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        self.extractor = LandmarkExtractor()
        self.boundary_detector = SentenceBoundaryDetector()
        self.speaker = Speaker()

        self.recognizer, self.gloss_classes = self._load_recognizer()
        self.seq2seq, self.gloss_seq_vocab, self.sentence_vocab = self._load_seq2seq()

        self.last_sentence = ""

    # ------------------------------------------------------------------
    def _load_recognizer(self):
        ckpt = torch.load(config.RECOGNIZER_CKPT, map_location=self.device)
        classes = ckpt["classes"]
        model = SignRecognizer(num_classes=len(classes)).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, classes

    def _load_seq2seq(self):
        gloss_seq_vocab_path = config.GLOSS_VOCAB_PATH.replace("gloss_vocab", "gloss_seq_vocab")
        gloss_seq_vocab = Vocab.load(gloss_seq_vocab_path)
        sentence_vocab = Vocab.load(config.SENTENCE_VOCAB_PATH)

        ckpt = torch.load(config.SEQ2SEQ_CKPT, map_location=self.device)
        model = Gloss2SentenceTransformer(
            gloss_vocab_size=ckpt["gloss_vocab_size"],
            sentence_vocab_size=ckpt["sentence_vocab_size"],
            pad_id=sentence_vocab.pad_id,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, gloss_seq_vocab, sentence_vocab

    # ------------------------------------------------------------------
    def recognize_glosses(self, utterance_frames):
        """Slide a window over the buffered utterance to produce a
        sequence of gloss predictions, then collapse consecutive
        duplicates (simple CTC-style de-duplication)."""
        arr = np.stack(utterance_frames, axis=0)  # (T, feature_dim)
        T = arr.shape[0]
        w, stride = config.SLIDING_WINDOW_SIZE, config.SLIDING_WINDOW_STRIDE

        raw_preds = []
        for start in range(0, max(1, T - w + 1), stride):
            window = arr[start:start + w]
            if window.shape[0] < w:
                pad = np.repeat(window[-1:], w - window.shape[0], axis=0)
                window = np.concatenate([window, pad], axis=0)
            window_t = torch.from_numpy(window).float().to(self.device)
            pred_id, confidence = self.recognizer.predict_with_confidence(window_t)
            predicted_class = self.gloss_classes[pred_id]
            if confidence >= config.GLOSS_CONFIDENCE_THRESHOLD and predicted_class != config.BACKGROUND_CLASS_NAME:
                raw_preds.append(predicted_class)

        # Collapse consecutive duplicate predictions (e.g. AAAB -> AB)
        glosses = []
        for g in raw_preds:
            if not glosses or glosses[-1] != g:
                glosses.append(g)
        return glosses

    def glosses_to_sentence(self, glosses):
        if not glosses:
            return ""
        src_ids = self.gloss_seq_vocab.encode(
            [g.lower() for g in glosses], max_len=config.MAX_GLOSS_LEN)
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        out_ids = self.seq2seq.greedy_decode(
            src, self.sentence_vocab.sos_id, self.sentence_vocab.eos_id)
        tokens = self.sentence_vocab.decode(out_ids)
        return detokenize(tokens)

    # ------------------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        print("Starting real-time translation. Press 'q' to quit.")
        prev_time = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    break
                frame = cv2.flip(frame, 1)

                feature_vec, results, hands_present = self.extractor.process(frame)
                frame = self.extractor.draw_landmarks(frame, results)

                utterance = self.boundary_detector.update(feature_vec, hands_present)
                if utterance is not None:
                    self._handle_utterance(utterance)

                frame = self._draw_hud(frame, prev_time)
                prev_time = time.time()

                cv2.imshow("Sign Language to Voice Translator", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.speaker.stop()
            self.extractor.close()
            cap.release()
            cv2.destroyAllWindows()

    def _handle_utterance(self, utterance_frames):
        glosses = self.recognize_glosses(utterance_frames)
        sentence = self.glosses_to_sentence(glosses)
        if sentence:
            print(f"Glosses: {glosses}  ->  Sentence: \"{sentence}\"")
            self.last_sentence = sentence
            self.speaker.say(sentence)

    def _draw_hud(self, frame, prev_time):
        fps = 1.0 / max(1e-6, time.time() - prev_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Last: {self.last_sentence}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame


if __name__ == "__main__":
    app = SignToVoiceTranslator()
    app.run()
