"""
collect_data.py
-----------------
Simple webcam-based data collection tool for building your own sign
dataset. For each gloss (word/sign) you want to teach the recognizer,
this records several short clips of landmark sequences and saves them
as .npy files under dataset/<GLOSS_NAME>/.

Usage:
    python -m data.collect_data --gloss HELLO --samples 30 --frames 24

Controls while running:
    SPACE  - start/stop recording a sample
    q      - quit

Each sample is a (frames, FRAME_FEATURE_DIM) numpy array, ready to be
consumed by data/dataset.py -> SignRecognitionDataset.
"""

import argparse
import os
import time
import numpy as np
import cv2

import config
from utils.landmark_extractor import LandmarkExtractor


def collect(gloss: str, num_samples: int, frames_per_sample: int, camera_index: int):
    out_dir = os.path.join(config.DATA_DIR, gloss.upper())
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    extractor = LandmarkExtractor()

    recording = False
    buffer = []
    sample_count = 0

    print(f"Collecting samples for gloss '{gloss}'. Press SPACE to start/stop "
          f"a recording, 'q' to quit. Target: {num_samples} samples of "
          f"{frames_per_sample} frames each.")

    try:
        while sample_count < num_samples:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = cv2.flip(frame, 1)

            feature_vec, results, hands_present = extractor.process(frame)
            frame = extractor.draw_landmarks(frame, results)

            if recording:
                buffer.append(feature_vec)
                cv2.putText(frame, f"REC {len(buffer)}/{frames_per_sample}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if len(buffer) >= frames_per_sample:
                    _save_sample(out_dir, existing + sample_count, buffer)
                    sample_count += 1
                    buffer = []
                    recording = False
                    print(f"Saved sample {sample_count}/{num_samples}")
            else:
                cv2.putText(frame, f"Gloss: {gloss} | Saved: {sample_count}/{num_samples} "
                                    "| SPACE=record  q=quit",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Sign Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not recording:
                    buffer = []
                recording = not recording
            elif key == ord("q"):
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. {sample_count} new samples saved to {out_dir}")


def _save_sample(out_dir, index, buffer):
    arr = np.stack(buffer, axis=0)  # (T, feature_dim)
    fname = f"sample_{index:04d}_{int(time.time())}.npy"
    np.save(os.path.join(out_dir, fname), arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect labeled sign landmark sequences.")
    parser.add_argument("--gloss", required=True, help="Gloss/word label for this recording session.")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to record.")
    parser.add_argument("--frames", type=int, default=config.SLIDING_WINDOW_SIZE,
                         help="Frames per recorded sample.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    collect(args.gloss, args.samples, args.frames, args.camera)
