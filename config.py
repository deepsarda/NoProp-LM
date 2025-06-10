# -----------------------------------------------------------------------------
# WandB & Experiment Tracking Configuration
# -----------------------------------------------------------------------------
WANDB_PROJECT_NAME = "NOPROP-LM"  # Project name for the main training runs
WANDB_RUN_NAME = "train_run"  # Default run name for main training
WANDB_PRETRAIN_RUN_NAME = "pretrain_run"  # Default run name for pretraining
SEED = 42  # Random seed for reproducibility


# -----------------------------------------------------------------------------
# Embedding Pretraining Configuration
# -----------------------------------------------------------------------------
# This section defines parameters for the optional embedding pretraining phase.
# Pretraining aims to create meaningful embeddings before the main NoProp training.
DO_PRETRAINING = True  # Master switch to enable or disable the pretraining phase
PRETRAINING_EPOCHS = 1  # Number of epochs for pretraining
PRETRAINING_LR = 1e-3  # Learning rate for pretraining
PRETRAINING_BATCH_SIZE = 640  # Batch size for pretraining
PRETRAINING_MASK_PROB = (
    0.15  # Probability of masking tokens during pretraining (similar to BERT MLM)
)
PRETRAINED_EMBEDDING_FILE = (
    "pretrained_embeddings.pt"  # File to save/load pretrained embeddings
)

# -----------------------------------------------------------------------------
#  Architecture Configuration
# -----------------------------------------------------------------------------
# This section specifies the architecture of the NoProp-LM model.
NUM_DENOISING_BLOCKS = 10  # Number of denoising blocks in the model
EMBEDDING_DIM = 384  # Dimensionality of token embeddings

# Configuration for each individual denoising block
BLOCK_DIM = EMBEDDING_DIM  # Input/output dimension of each block (usually same as EMBEDDING_DIM)
BLOCK_NHEAD = 4  # Number of attention heads in each block's multi-head attention layer
BLOCK_NUM_ENCODER_LAYERS = 1  # Number of transformer encoder layers within each block
BLOCK_NUM_DECODER_LAYERS = 1  # Number of transformer decoder layers within each block
BLOCK_NUM_PRETRAIN_LAYERS = 3  # Number of pretraining layers
BLOCK_DIM_FEEDFORWARD = (
    BLOCK_DIM * 7 // 2
)  # Dimension of the feed-forward network in each block.
# Normally 4 * BLOCK_DIM, but 3.5 * BLOCK_DIM is used here for fewer parameters.
BLOCK_DROPOUT_RATE = 0.1  # Dropout rate used within blocks
LAYER_NORM_EPS = 1e-5  # Epsilon for layer normalization to prevent division by zero

# -----------------------------------------------------------------------------
# Dataset & Tokenization Configuration
# -----------------------------------------------------------------------------
# This section defines the dataset and tokenizer to be used.
DATASET_NAME = (
    "wikitext"  # Name of the dataset (e.g., from Hugging Face datasets library)
)
DATASET_CONFIG_NAME = "wikitext-103-raw-v1"  # Specific configuration of the dataset
TOKENIZER_NAME = "bert-base-uncased"  # Name of the tokenizer (e.g., from Hugging Face tokenizers library)
MAX_SEQ_LENGTH = 128  # Maximum sequence length for tokenized inputs
IGNORE_INDEX = (
    -100
)  # Index to ignore in loss calculation (typically for padding tokens)

# -----------------------------------------------------------------------------
# NOPROP Training Configuration
# -----------------------------------------------------------------------------
# This section contains parameters for the main NoProp training phase.
EPOCHS = 3  # Number of epochs for NoProp training
BATCH_SIZE = 1280  # Batch size for NoProp training
LEARNING_RATE = 1e-4  # Learning rate for NoProp training
FP16_ENABLED = True  # Enable mixed-precision training (uses NVIDIA's AMP)
GRAD_CLIP_VALUE = 1.0  # Value for gradient clipping to prevent exploding gradients
ADAM_BETAS = (0.9, 0.98)  # Beta parameters for the Adam optimizer
ADAM_EPS = 1e-5  # Epsilon parameter for the Adam optimizer
DATALOADER_NUM_WORKERS_TRAIN = (
    4  # Number of worker processes for the training DataLoader
)

# -----------------------------------------------------------------------------
# Evaluation & Generation Configuration
# -----------------------------------------------------------------------------
# This section configures how the model is evaluated and how text generation is performed during training.
LOG_GENERATION_EVERY_N_STEPS = 1500  # Log generated text samples every N training steps
NUM_GENERATION_EXAMPLES_TO_LOG = 3  # Number of text generation examples to log
GENERATION_MAX_OUTPUT_LENGTH = (
    16  # Maximum length of generated sequences during logging
)
TEST_PROMPTS_FOR_LOGGING = [  # List of prompts to use for generating sample text
    "The weather is",
    "Once upon a time, in a land far away,",
    "The sky is",
]

# -----------------------------------------------------------------------------
# Checkpointing & Saving Configuration
# -----------------------------------------------------------------------------
# This section defines how model checkpoints and the final model are saved.
FINAL_MODEL_SAVE_PATH = "noprop_lm.pt"  # Path to save the final trained model's weights

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------
# This section specifies hardware-related configurations.
DEVICE = "cuda"  # Device to use for training (e.g., "cuda" for GPU, "cpu" for CPU)
