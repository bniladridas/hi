# Token Classification Model

[![CI](https://github.com/yourusername/token-classification-ai-fine-tune/workflows/CI/badge.svg)](https://github.com/yourusername/token-classification-ai-fine-tune/actions)
[![codecov](https://codecov.io/gh/yourusername/token-classification-ai-fine-tune/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/token-classification-ai-fine-tune)

Fine-tunes BERT for Named Entity Recognition (NER) on CoNLL-2003 dataset.

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run
docker build -t token-classification-model .
docker run --rm token-classification-model

# Or use the convenience script
./scripts/run_model.sh
```

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training
python src/model.py

# Run tests
pytest
```

## Development

### Code Quality

This project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Testing

```bash
# Run unit tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Docker Development

```bash
# Run tests in container
docker-compose run --rm model-testing

# Start Jupyter notebook
docker-compose up jupyter

# Train model
docker-compose run --rm model-training
```

## CI/CD

The project includes comprehensive CI/CD:

- **Pre-commit checks**: Code formatting, linting, and style checks
- **Unit tests**: pytest with coverage reporting
- **Docker build**: Ensures container builds correctly
- **Model validation**: Basic functionality tests
- **Codecov integration**: Coverage reporting

### CI Pipeline

- Runs on every push/PR to main branch
- Tests across Python 3.12
- Docker container validation
- Pre-commit hook validation

## Project Structure

```
├── src/
│   ├── __init__.py          # Package initialization
│   └── model.py             # Main model training code
├── tests/
│   ├── __init__.py
│   └── test_model.py        # Unit tests
├── scripts/
│   ├── commit-msg           # Git commit message validation
│   ├── rewrite_msg.sh       # Commit message utilities
│   └── run_model.sh         # Convenience script
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI
├── Dockerfile              # Container definition
├── docker-compose.yml      # Local development setup
├── requirements.txt        # Python dependencies
├── VERSION                 # Version tracking
└── .pre-commit-config.yaml # Code quality hooks
```

## Model Details

- **Architecture**: BERT-base-uncased
- **Task**: Named Entity Recognition (NER)
- **Dataset**: CoNLL-2003
- **Labels**: 9 NER tags (O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC)

## Inference

After training, use the model for inference:

```python
from transformers import pipeline

# Load from Hugging Face Hub
ner = pipeline("token-classification", model="harpertoken/harpertokenNER")
result = ner("Apple is buying a U.K. startup for $1 billion")
print(result)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure CI passes
5. Submit a pull request

## License

Licensed under Apache 2.0.
