"""
collect_evaluation_data.py
--------------------------
Collect separate validation and test samples for the sign-language
recognition model.

This script does NOT save samples to the training dataset.

Examples:
    python -m data.collect_evaluation_data --split validation --gloss HOW --samples 2
    python -m data.collect_evaluation_data --split test --gloss HOW --samples 20

Controls:
    SPACE - start recording a sample
    Q     - quit

Each sample is saved as a NumPy array with shape:
    (frames, config.FRAME_FEATURE_DIM)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config
from utils.landmark_extractor import LandmarkExtractor


VALID_CLASSES = {"BACKGROUND", "FINE", "HOW", "I", "YOU"}


def collect(
    split: str,
    gloss: str,
    num_samples: int,
    frames_per_sample: int,
    camera_index: int,
):
    # Store evaluation data outside the training dataset.
    project_dir = Path(config.BASE_DIR)
    output_dir = project_dir / "evaluation_data" / split / gloss
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting existing samples.
    existing_files = list(output_dir.glob("*.npy"))
    next_sample_number = len(existing_files) + 1

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
        print("Check that your webcam is connected and not being used")
        print("by another application.")
        return

    extractor = None
    recording = False
    buffer = []
    collected = 0

    try:
        extractor = LandmarkExtractor()

        print("\n========================================")
        print(" SIGN LANGUAGE EVALUATION DATA COLLECTOR")
        print("========================================")
        print(f"Split:           {split}")
        print(f"Class:           {gloss}")
        print(f"Target samples:  {num_samples}")
        print(f"Frames/sample:   {frames_per_sample}")
        print(f"Feature count:   {config.FRAME_FEATURE_DIM}")
        print(f"Save directory:  {output_dir}")
        print("\nPress SPACE to start recording one sample.")
        print("Keep signing until recording finishes.")
        print("Press Q to quit.")
        print("========================================\n")

        while collected < num_samples:
            ok, frame = cap.read()

            if not ok:
                print("ERROR: Could not read a frame from the webcam.")
                break

            # Mirror the camera image for easier signing.
            frame = cv2.flip(frame, 1)

            # Extract features exactly once per frame.
            features, results, hands_present = extractor.process(frame)

            # Draw MediaPipe landmarks for visual feedback.
            frame = extractor.draw_landmarks(frame, results)

            if recording:
                buffer.append(features)

                cv2.putText(
                    frame,
                    f"RECORDING: {len(buffer)}/{frames_per_sample}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"Class: {gloss} | Split: {split}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Saved: {collected}/{num_samples}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "SPACE = record sample | Q = quit",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # Save once the required number of frames is collected.
            if recording and len(buffer) >= frames_per_sample:
                sample = np.stack(
                    buffer[:frames_per_sample],
                    axis=0,
                ).astype(np.float32)

                expected_shape = (
                    frames_per_sample,
                    config.FRAME_FEATURE_DIM,
                )

                if sample.shape != expected_shape:
                    print(
                        f"ERROR: Unexpected sample shape {sample.shape}; "
                        f"expected {expected_shape}. Sample not saved."
                    )
                else:
                    filename = (
                        f"sample_{next_sample_number:04d}_"
                        f"{int(time.time())}.npy"
                    )
                    save_path = output_dir / filename

                    np.save(save_path, sample)

                    next_sample_number += 1
                    collected += 1

                    print(
                        f"Saved {collected}/{num_samples}: {save_path.name}"
                    )

                buffer = []
                recording = False

            cv2.imshow("Evaluation Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Collection stopped by user.")
                break

            elif key == ord(" ") and not recording:
                buffer = []
                recording = True
                print("Recording started...")

    except Exception as exc:
        print(f"\nERROR during data collection: {exc}")
        raise

    finally:
        if extractor is not None:
            extractor.close()

        cap.release()
        cv2.destroyAllWindows()

    print("\nCollection session finished.")
    print(f"New samples collected: {collected}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect separate sign-recognition evaluation data."
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["validation", "test"],
        help="Choose validation or test data.",
    )

    parser.add_argument(
        "--gloss",
        required=True,
        type=str.upper,
        choices=sorted(VALID_CLASSES),
        help="Class to record.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to record.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=config.SLIDING_WINDOW_SIZE,
        help="Number of frames in each sample.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Webcam index.",
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1.")

    if args.frames < 1:
        parser.error("--frames must be at least 1.")

    collect(
        split=args.split,
        gloss=args.gloss,
        num_samples=args.samples,
        frames_per_sample=args.frames,
        camera_index=args.camera,
    )


if __name__ == "__main__":
    main()

