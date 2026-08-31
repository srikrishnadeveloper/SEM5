# config.py - all the settings for the filament project
import os
import torch

_env = os.environ.get
_flag = lambda name, default="0": _env(name, default).lower() in {"1", "true", "yes", "on"}
_csv_float = lambda name, default: [float(x) for x in _env(name, default).split(",") if x.strip()]
_csv_int = lambda name, default: [int(x) for x in _env(name, default).split(",") if x.strip()]

BASE_PATH = _env("FILAMENT_BASE_PATH", r"C:\Users\srik2\Desktop\College\Machine Learning\Kaggle_Competition_Mini_Project\data\MAGFiLO_1.0_Kaggle_2026")
TRAIN_DIR, TEST_DIR = os.path.join(BASE_PATH, "train"), os.path.join(BASE_PATH, "test")
TRAIN_IMAGES, TEST_IMAGES = os.path.join(TRAIN_DIR, "train_images"), os.path.join(TEST_DIR, "test_images")
TRAIN_JSON = os.path.join(TRAIN_DIR, "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "code", "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")
for _d in (MODELS_DIR, PLOTS_DIR, SUBMISSIONS_DIR): os.makedirs(_d, exist_ok=True)

IMG_SIZE = 2048
TRAIN_RES = int(_env("FILAMENT_TRAIN_RES", "1024"))
INFERENCE_RES = int(_env("FILAMENT_INFERENCE_RES", str(IMG_SIZE)))
ENCODER = _env("FILAMENT_ENCODER", "tu-efficientnet_b4")
MODEL_NAME = _env("FILAMENT_MODEL_NAME", "unetplusplus")
IN_CHANNELS = int(_env("FILAMENT_IN_CHANNELS", "3"))
CLASSES = 1
PRETRAINED = _flag("FILAMENT_PRETRAINED", "1")
DECODER_USE_BATCHNORM = _env("FILAMENT_DECODER_USE_BATCHNORM", "inplace")
GRADIENT_CHECKPOINTING = _flag("FILAMENT_GRADIENT_CHECKPOINTING", "1")
DEEP_SUPERVISION = _flag("FILAMENT_DEEP_SUPERVISION", "0")

SEED = int(_env("FILAMENT_SEED", "2026"))
BATCH_SIZE = int(_env("FILAMENT_BATCH_SIZE", "1"))
EPOCHS = int(_env("FILAMENT_EPOCHS", "50"))
LR = float(_env("FILAMENT_LR", "1e-4"))
MIN_LR = float(_env("FILAMENT_MIN_LR", "1e-6"))
WD = float(_env("FILAMENT_WD", "1e-5"))
PATIENCE = int(_env("FILAMENT_PATIENCE", "10"))
ACCUMULATION_STEPS = int(_env("FILAMENT_ACCUMULATION", "8"))
NUM_WORKERS = int(_env("FILAMENT_NUM_WORKERS", "2"))
USE_AMP = _flag("FILAMENT_USE_AMP", "1")
USE_EMA = _flag("FILAMENT_USE_EMA", "1")
EMA_DECAY = float(_env("FILAMENT_EMA_DECAY", "0.999"))
USE_WARMUP = _flag("FILAMENT_USE_WARMUP", "1")
WARMUP_EPOCHS = int(_env("FILAMENT_WARMUP_EPOCHS", "5"))
RESUME = _env("FILAMENT_RESUME", "")
CHECKPOINT_EVERY = int(_env("FILAMENT_CHECKPOINT_EVERY", "5"))

LOSS_DICE_W = float(_env("FILAMENT_LOSS_DICE_W", "0.3"))
LOSS_FOCAL_W = float(_env("FILAMENT_LOSS_FOCAL_W", "0.0"))
LOSS_TVERSKY_W = float(_env("FILAMENT_LOSS_TVERSKY_W", "0.4"))
LOSS_BOUNDARY_W = float(_env("FILAMENT_LOSS_BOUNDARY_W", "0.1"))
LOSS_BCE_W = float(_env("FILAMENT_LOSS_BCE_W", "0.2"))
POS_WEIGHT = float(_env("FILAMENT_POS_WEIGHT", "0"))
FOCAL_GAMMA, FOCAL_ALPHA = 2.0, 0.25

N_FOLDS = int(_env("FILAMENT_N_FOLDS", "5"))
VAL_FOLD = int(_env("FILAMENT_VAL_FOLD", "0"))
VAL_METRIC = _env("FILAMENT_VAL_METRIC", "mIoU_multiscale")
MASK_AWARE_CROP = _flag("FILAMENT_MASK_AWARE_CROP", "1")
HEAVY_AUG = _flag("FILAMENT_HEAVY_AUG", "1")
COPY_PASTE = _flag("FILAMENT_COPY_PASTE", "0")
MULTI_SCALE_TRAIN = _flag("FILAMENT_MULTI_SCALE_TRAIN", "0")

PROB_THRESHOLD = float(_env("FILAMENT_PROB_THRESHOLD", "0.45"))
MORPH_CLOSE_K = int(_env("FILAMENT_MORPH_CLOSE_K", "3"))
MIN_AREA = int(_env("FILAMENT_MIN_AREA", "200"))
MAX_AREA = int(_env("FILAMENT_MAX_AREA", "500000"))
TTA_D4 = _flag("FILAMENT_TTA_D4", "0")
TTA_SCALES = _csv_float("FILAMENT_TTA_SCALES", "0.8,1.0,1.25")
CRF = _flag("FILAMENT_CRF", "0")
CRF_ITER = int(_env("FILAMENT_CRF_ITER", "5"))
OOF_THRESHOLD_MIN = float(_env("FILAMENT_OOF_THRESHOLD_MIN", "0.30"))
OOF_THRESHOLD_MAX = float(_env("FILAMENT_OOF_THRESHOLD_MAX", "0.75"))
OOF_THRESHOLD_STEP = float(_env("FILAMENT_OOF_THRESHOLD_STEP", "0.05"))
OOF_MIN_AREAS = _csv_int("FILAMENT_OOF_MIN_AREAS", "50,100,200,500,1000")
NO_FILAMENT_PROB = float(_env("FILAMENT_NO_FILAMENT_PROB", "0.10"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
