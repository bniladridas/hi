from unittest.mock import patch, MagicMock

from src.model import tokenize_and_align_labels


class MockTokenizedInputs(dict):
    def word_ids(self, batch_index):
        return [None, 0, 1, None]

@patch("src.model.tokenizer")
def test_tokenize_and_align_labels(mock_tokenizer):
    mock_result = MockTokenizedInputs({
        'input_ids': [[101, 102, 103, 102]],
        'attention_mask': [[1, 1, 1, 1]],
    })
    mock_tokenizer.return_value = mock_result

    examples = {"tokens": [["Hello", "world"]], "ner_tags": [[0, 1]]}

    result = tokenize_and_align_labels(examples)

    assert "labels" in result
    assert result["labels"] == [[-100, 0, 1, -100]]
