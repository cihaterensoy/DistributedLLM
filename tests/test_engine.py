"""
Otomatik pytest testleri.
İnternetten model indirmeden, bellekte anlık mini bir yapay model üreterek motoru test eder.
"""
import pytest
import torch
from transformers import GPT2Config

from src.distributed_llm.engine import manual_gpt2_logits, layer_report


def test_manual_logits_match_reference_without_network():
    transformers = pytest.importorskip('transformers')
    GPT2Config=transformers.GPT2Config
    GPT2LMHeadModel=transformers.GPT2LMHeadModel

    torch.manual_seed(42)

    config=GPT2Config(
        vocab_size=101,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0
    )
    # Konfigürasyona uygun rastgele ağırlıklı modeli CPU'da oluşturuyoruz
    model = GPT2LMHeadModel(config).eval()
    input_ids = torch.tensor([[1, 5, 9, 7]])

    with torch.inference_mode():
        # Beklenen sonuç
        expected = model(input_ids, use_cache=False).logits
        # Bizim fonksiyonumuzun sonucu
        actual = manual_gpt2_logits(model, input_ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5) # İkisi tolerans dahilinde birebir aynı mı?

    assert len(layer_report(model)) == 2 # Katman raporunun gerçekten 2 katman döndüğünü doğruluyoruz

def test_bad_input_shape():
    with pytest.raises(ValueError, match="shape"):
        manual_gpt2_logits(None, torch.tensor([1, 2, 3]))