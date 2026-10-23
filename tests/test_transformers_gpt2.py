"""Gerçek Hugging Face GPT-2 ile ağ erişimsiz uçtan uca test.

transformers kurulmadıysa skip olur; tam doğrulama için skip olmamalıdır.
"""
import pytest
import torch


def test_real_gpt2_save_pretrained_streaming(tmp_path):
    transformers = pytest.importorskip('transformers')
    from src.distributed_llm.streaming import StreamingCausalLM
    torch.manual_seed(4)
    config = transformers.GPT2Config(
        vocab_size=101, bos_token_id=0, eos_token_id=1,
        n_positions=32, n_ctx=32, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    )
    reference = transformers.GPT2LMHeadModel(config).eval()
    reference.save_pretrained(tmp_path, safe_serialization=True)
    token_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        expected = reference(token_ids, use_cache=False).logits
    del reference
    streaming = StreamingCausalLM(tmp_path)
    actual = streaming.logits(token_ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert streaming.resident_layer_count() == 0
    assert streaming.layer_reads == 2
