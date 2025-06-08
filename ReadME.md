
# NoProp-LM: A Language Model Trained Without End-to-End Backpropagation

This repository contains a PyTorch implementation of a generative language model based on the principles from the paper ["NOPROP: TRAINING NEURAL NETWORKS WITHOUT BACK-PROPAGATION OR FORWARD-PROPAGATION"](https://arxiv.org/abs/2305.17333).

This project adapts the core NoProp methodology from image classification to the domain of autoregressive language modeling. Instead of training a deep, monolithic network with backpropagation, we train a series of independent, shallow "denoising" blocks. Each block is an expert at cleaning up a "noisy" or "fuzzy" word embedding at a specific noise level, conditioned on the preceding text context.

The result is a training paradigm that explores alternatives to traditional deep learning techniques. Allowing for the training of even larger language models.

*Note: The code in this repositery has been designed to train a LM on single T4.*
## Table of Contents
- [Architecture](#architecture)
- [Advantages vs. Disadvantages](#advantages-vs-disadvantages)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Configuration](#configuration)

## Architecture

The model consists of two main components:

1.  **Frozen Embedding Table:** A standard `nn.Embedding` layer that is pre-trained using a Masked Language Modeling (MLM) objective. After pre-training, its weights are frozen. This table serves as the "ground truth" for clean token embeddings.

2.  **A Chain of `DenoisingBlock`s:**
    *   There are `N` (e.g., 12) independent `DenoisingBlock` modules.
    *   Each block is a complete, self-contained **Causal Transformer Encoder**.
    *   During training, each block is trained on its own local loss: to denoise a sequence of noisy target embeddings back to their clean state, conditioned on a sequence of clean context embeddings.
    *   The noise level is determined by a cosine schedule, where earlier blocks handle more noise and later blocks handle less.

## Advantages vs. Disadvantages

This training paradigm presents a unique set of trade-offs compared to traditional, backpropagation-based deep learning models.

### Advantages

1.  **Reduced Memory Usage During Training:** Since only one shallow `DenoisingBlock` is loaded onto the GPU at a time, the peak memory requirement is significantly lower — approx `1/N` times, where is the number of blocks — than training a deep, end-to-end model of equivalent depth. There is no need to store intermediate activations for a deep backward pass.

2.  **Massive Parallelism Potential:** Because the blocks are independent, their training can be fully parallelized across multiple GPUs — or even multiple machines — without requiring complex synchronization mechanisms like parameter servers. For example, Block #1 could be trained on GPU-1, Block #2 on GPU-2, and so on, simultaneously. This approach has the potential to enable highly decentralized model training, where each block can be trained in a different data center around the world without needing communication between individual data centers.

3.  **No "Update Locking":** In standard backprop, the weights of early layers cannot be updated until the error signal has propagated all the way back from the final layer. NoProp removes this dependency, allowing for more flexible and decoupled training schedules.

### Disadvantages

1.  **Slower Training Time (on a single GPU):** The current implementation trains the blocks sequentially. This means the dataset must be iterated through `N` times per epoch, where `N` is the number of blocks. This is much slower than a single pass through a standard model on one GPU. *Note: This disadvantage disappears if training is parallelized across multiple devices.*

2.  **No Joint Optimization:** The layers do not learn to cooperate in a jointly optimized way. Block #3 has no direct influence on how Block #2 learns. This may lead to a final model that is less performant than a jointly-optimized network of the same size, as subtle, co-adapted features cannot be learned.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/deepsarda/NoProp-LM
    cd NoProp-LM
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Tokenizer:**
    The default configuration uses bert-base-uncased. To run locally, you can change `TOKENIZER_NAME` in `config.py` to a standard Hugging Face tokenizer, e.g., `"bert-base-uncased"`.

## How to Run

The entire training process is orchestrated by the `run_training.py` script.

```bash
python run_training.py
```

This script will automatically perform two phases:
1.  **Phase 1 (Pre-training):** It will train the `MLMPretrainModel` on the specified dataset to learn the token embeddings and save them to `pretrained_embeddings.pt`. This can be disabled in the config.
2.  **Phase 2 (NoProp Training):** It will initialize the main `LanguageModel`, load the frozen embeddings, and then execute the NoProp training loop, training each `DenoisingBlock` sequentially on the full dataset.

## Configuration

All important settings can be modified in `config.py`. Key options include:

*   `DO_PRETRAINING`: Set to `False` to skip the pre-training phase if you already have `pretrained_embeddings.pt`.
*   `NUM_DENOISING_BLOCKS`: The number of specialists in the chain.
*   `EMBEDDING_DIM`, `BLOCK_NHEAD`, etc.: Control the model's dimensions.
*   `DATASET_NAME`: Specify which Hugging Face dataset to use.
*   `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: Standard training hyperparameters.
*   `LOG_GENERATION_EVERY_N_STEPS`: How often to log generated text samples.