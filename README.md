# Token Classification Model

A fine-tuned BERT model for Named Entity Recognition on the CoNLL-2003 dataset.

## Features

- Optimized for NVIDIA GPUs with mixed precision
- TensorBoard integration for logging
- Available on Hugging Face Hub

## Usage

```python
from transformers import pipeline

ner = pipeline("token-classification", model="bniladridas/token-classification-ai-fine-tune")
print(ner("Apple is buying a U.K. startup for $1 billion"))
```

## Training Details

- Dataset: CoNLL-2003
- Learning Rate: 2e-05
- Batch Size: 16
- Epochs: 3
- Training Loss: 0.0160
- Validation Loss: 0.0474

## Links

- [Model on Hugging Face](https://huggingface.co/bniladridas/token-classification-ai-fine-tune)
- [CoNLL-2003 Dataset](https://huggingface.co/datasets/conll2003)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)

## Requirements

- NVIDIA GPU + CUDA
- PyTorch 2.0.1+
- Transformers 4.28.1+

## Conventional Commits

This project uses conventional commit standards.

### Setup

To enable the commit message hook:

```bash
cp scripts/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

### Usage

Commit messages must start with types like `feat:`, `fix:`, etc., followed by a lowercase description ≤40 characters.

Allowed types: feat, fix, docs, style, refactor, test, chore

### History Cleanup

To rewrite existing commit messages (lowercase + truncate to 40 chars):

```bash
bash scripts/rewrite_msg.sh
```

Force push after rewriting if you have a remote.

Licensed under Apache 2.0. Tags: `token-classification`, `conll2003`, `generated_from_trainer`.
