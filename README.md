# Token Classification Model

Fine-tunes BERT for NER on CoNLL-2003 dataset.

## Run

```bash
python src/model.py
```

## Inference

```python
from transformers import pipeline
ner = pipeline("token-classification", model="bniladridas/token-classification-ai-fine-tune")
print(ner("Apple is buying a U.K. startup for $1 billion"))
```

Licensed under Apache 2.0.
