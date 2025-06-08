# -----------------------------------------------------------------------------
# WandB & Experiment Tracking Configuration
# -----------------------------------------------------------------------------
WANDB_PROJECT_NAME = "NOPROP-LM-TRAIN"
WANDB_PRETRAIN_PROJECT_NAME = "NOPROP-LM-PRETRAIN"  # Separate project for pretraining
WANDB_RUN_NAME = "noprop_lm_run"
WANDB_PRETRAIN_RUN_NAME = "embedding_pretrain_run"
SEED = 42


# -----------------------------------------------------------------------------
# Embedding Pretraining Configuration
# -----------------------------------------------------------------------------
DO_PRETRAINING = True  # Master switch to run the pretraining phase
PRETRAINING_EPOCHS = 1
PRETRAINING_LR = 3e-3
PRETRAINING_BATCH_SIZE = 640
PRETRAINING_MASK_PROB = 0.15
PRETRAINED_EMBEDDING_FILE = "pretrained_embeddings.pt"

# -----------------------------------------------------------------------------
#  Architecture Configuration
# -----------------------------------------------------------------------------
NUM_DENOISING_BLOCKS = 10
EMBEDDING_DIM = 384

BLOCK_DIM = EMBEDDING_DIM
BLOCK_NHEAD = 4
BLOCK_NUM_ENCODER_LAYERS = 3
BLOCK_DIM_FEEDFORWARD = (
    BLOCK_DIM * 7 // 2
)  # Normally 4 * BLOCK_DIM, but 3.5 * BLOCK_DIM works well for less parameters. FF is the slowest part of the model.
BLOCK_DROPOUT_RATE = 0.1
LAYER_NORM_EPS = 1e-5

NOISE_SCHEDULE_TYPE = "cosine"

# -----------------------------------------------------------------------------
# Dataset & Tokenization Configuration
# -----------------------------------------------------------------------------
DATASET_NAME = "wikitext"
DATASET_CONFIG_NAME = "wikitext-103-raw-v1"
TOKENIZER_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 128
IGNORE_INDEX = -100

# -----------------------------------------------------------------------------
# NOPROP Training Configuration
# -----------------------------------------------------------------------------
EPOCHS = 3
BATCH_SIZE = 640
LEARNING_RATE = 5e-4
FP16_ENABLED = True
GRAD_CLIP_VALUE = 1.0
ADAM_BETAS = (0.9, 0.98)
ADAM_EPS = 1e-5

DATALOADER_NUM_WORKERS_TRAIN = 4

# -----------------------------------------------------------------------------
# Evaluation & Generation Configuration
# -----------------------------------------------------------------------------
LOG_GENERATION_EVERY_N_STEPS = 1500
NUM_GENERATION_EXAMPLES_TO_LOG = 3
GENERATION_MAX_OUTPUT_LENGTH = 16
TEST_PROMPTS_FOR_LOGGING = [
    "The weather is",
    "Once upon a time, in a land far away,",
    "The sky is",
]

# -----------------------------------------------------------------------------
# Checkpointing & Saving Configuration
# -----------------------------------------------------------------------------
FINAL_MODEL_SAVE_PATH = "noprop_lm.pt"  # Path to save the final model's weights

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda"
