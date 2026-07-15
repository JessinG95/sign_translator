"""
config.py
---------
Central configuration for the Sign Language -> Voice Translator.
Keeping all tunable constants in one place makes the system easy to adjust
without hunting through every module.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")            # raw collected landmark sequences
RECOGNIZER_CKPT = os.path.join(BASE_DIR, "checkpoints", "sign_recognizer.pt")
SEQ2SEQ_CKPT = os.path.join(BASE_DIR, "checkpoints", "gloss2sentence.pt")
GLOSS_VOCAB_PATH = os.path.join(BASE_DIR, "checkpoints", "gloss_vocab.json")
SENTENCE_VOCAB_PATH = os.path.join(BASE_DIR, "checkpoints", "sentence_vocab.json")

os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# MediaPipe / landmark extraction
# ----------------------------------------------------------------------
# Landmark counts (MediaPipe Holistic)
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21          # per hand
NUM_FACE_LANDMARKS = 468

# We usually don't need all 468 face points for sign language (mouth shape,
# eyebrows etc. carry grammatical meaning, but the full mesh is overkill).
# We instead sample a small, meaningful subset of face points.
FACE_LANDMARK_SUBSET = list(range(0, 468, 14))  # ~33 sampled points, evenly spaced

# Feature vector size per frame = pose(x,y,z,vis) + 2*hand(x,y,z) + face_subset(x,y)
POSE_FEATS = NUM_POSE_LANDMARKS * 4
HAND_FEATS = NUM_HAND_LANDMARKS * 3 * 2
FACE_FEATS = len(FACE_LANDMARK_SUBSET) * 2
FRAME_FEATURE_DIM = POSE_FEATS + HAND_FEATS + FACE_FEATS

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ----------------------------------------------------------------------
# Sentence boundary / pause detection
# ----------------------------------------------------------------------
# Number of consecutive frames with "no hands visible" (or near-zero hand
# motion) required before we treat it as the end of a signed utterance.
PAUSE_FRAMES_THRESHOLD = 20        # ~0.7s at 30fps
MOTION_EPSILON = 0.015             # movement below this counts as "still"
MAX_UTTERANCE_FRAMES = 150         # hard cap (~5s) so long signing doesn't buffer forever
MIN_UTTERANCE_FRAMES = 8  
MIN_UTTERANCE_MOTION = 0.3         # ignore accidental blips shorter than this

# ----------------------------------------------------------------------
# Sign recognizer model (CNN + BiLSTM)
# ----------------------------------------------------------------------
RECOGNIZER_HIDDEN_DIM = 256
RECOGNIZER_LSTM_LAYERS = 2
RECOGNIZER_CNN_CHANNELS = 128
RECOGNIZER_DROPOUT = 0.3
SLIDING_WINDOW_SIZE = 24            # frames per gloss-recognition window
SLIDING_WINDOW_STRIDE = 8
GLOSS_CONFIDENCE_THRESHOLD = 0.55
BACKGROUND_CLASS_NAME = "BACKGROUND"

# ----------------------------------------------------------------------
# Seq2Seq Transformer (gloss sequence -> natural sentence)
# ----------------------------------------------------------------------
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FFN_DIM = 512
SEQ2SEQ_DROPOUT = 0.1
MAX_GLOSS_LEN = 32
MAX_SENTENCE_LEN = 32

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SEQ2SEQ_LEARNING_RATE = 3e-4
EPOCHS_RECOGNIZER = 50
EPOCHS_SEQ2SEQ = 100
DEVICE = "cuda"  # falls back to cpu automatically in code if unavailable

# ----------------------------------------------------------------------
# Text-to-Speech
# ----------------------------------------------------------------------
TTS_ENGINE = "pyttsx3"   # "pyttsx3" (offline) or "gtts" (online, needs internet)
TTS_RATE = 175
TTS_VOLUME = 1.0

# ----------------------------------------------------------------------
# Webcam
# ----------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
TARGET_FPS = 30
