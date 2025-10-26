import pytest
from unittest.mock import patch, MagicMock
from src.model import tokenize_and_align_labels


@patch('src.model.tokenizer')
def test_tokenize_and_align_labels(mock_tokenizer):
    # Mock tokenizer
    mock_tokenizer.return_value = {
        'input_ids': [[101, 102, 103, 102], [101, 104, 105, 102]],
        'attention_mask': [[1, 1, 1, 1], [1, 1, 1, 1]],
        'word_ids': lambda batch_index: [None, 0, 1, None] if batch_index == 0 else [None, 0, 1, None]
    }
    mock_tokenizer.word_ids = mock_tokenizer.return_value['word_ids']

    examples = {
        "tokens": [["Hello", "world"]],
        "ner_tags": [[0, 1]]
    }

    result = tokenize_and_align_labels(examples)

    assert "labels" in result
    assert result["labels"] == [[-100, 0, 1, -100]]