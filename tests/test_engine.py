import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from src.distributed_llm.engine import manual_gpt2_logits, layer_report


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=101, n_positions=32, n_ctx=32,
        n_embd=32, n_layer=2, n_head=4,
        bos_token_id=0, eos_token_id=1,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(config).eval()


@pytest.mark.parametrize("input_ids", [
    [[1, 5, 9, 7]],                  # Tekli batch
    [[1, 5, 9, 7], [3, 2, 8, 4]],   # Çoklu batch (batch_size = 2)
    [[1]],                           # Tek bir token
])
def test_manual_logits_match_reference(model, input_ids):
    tokens = torch.tensor(input_ids, dtype=torch.long)
    with torch.inference_mode():
        expected = model(tokens, use_cache=False).logits
        actual = manual_gpt2_logits(model, tokens)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_layer_report(model):
    report = layer_report(model)
    assert len(report) == 2
    assert all(row["parameters"] > 0 for row in report)


def test_bad_input_shape():
    with pytest.raises(ValueError, match="shape"):
        manual_gpt2_logits(None, torch.tensor([1, 2, 3]))


def test_sequence_too_long(model):
    """Modelin n_positions kapasitesinden (32) daha uzun dizi gelirse durdurmalı."""
    with pytest.raises(ValueError, match="position capacity"):
        manual_gpt2_logits(model, torch.ones((1, 33), dtype=torch.long))