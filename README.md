# Token Classification Model

Fine-tunes BERT for Named Entity Recognition (NER) on CoNLL-2003 dataset.

## Quick Start

### Using Docker

```bash
# Build and run
docker build -t token-classification-model .
docker run --rm token-classification-model

# Or use the convenience script
./scripts/run_model.sh
```

### Local Development

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]

# Run training
python src/model.py

# Run tests
pytest
```

## Development

### Code Quality

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Testing

```bash
pytest --cov=src --cov-report=html
```

### Docker Development

```bash
# Run tests in container
docker-compose run --rm model-testing

# Train model
docker-compose run --rm model-training
```

## CI/CD

- Pre-commit checks
- Unit tests with coverage
- Docker build validation
- Model validation

## Project Structure

```
├── src/                 # Source code
├── tests/              # Unit tests
├── scripts/            # Utilities
├── .github/workflows/  # CI pipeline
├── Dockerfile          # Container
├── docker-compose.yml  # Local dev
├── pyproject.toml      # Modern Python packaging
└── VERSION            # Version tracking
```

## Model Details

- **Architecture**: BERT-base-uncased
- **Task**: Named Entity Recognition
- **Dataset**: CoNLL-2003
- **Labels**: 9 NER tags

## Pushing to Hugging Face Hub

To push the trained model to Hugging Face Hub:

1. Set your Hugging Face token as an environment variable:
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```

2. Modify `src/model.py` to enable pushing:
   - Change `push_to_hub=False` to `push_to_hub=True`
   - Add `hub_model_id="harpertoken/harpertokenNER"`

3. Run the training script.

The model will be available at: https://huggingface.co/harpertoken/harpertokenNER

## Inference

```python
from transformers import pipeline
ner = pipeline("token-classification", model="harpertoken/harpertokenNER")
print(ner("Apple is buying a U.K. startup for $1 billion"))
```

## License

Apache 2.0
