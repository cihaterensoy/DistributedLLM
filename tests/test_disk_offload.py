import pytest
import torch
from safetensors.torch import load_file

from src.distributed_llm.disk_offload import _save_tensors, DiskOffloadedGPT2, export_gpt2_shards


def test_safetensors_roundtrip(tmp_path):
    """safetensors kütüphanesinin diskten okuma-yazma yapabildiğini basit bir matrisle test eder."""
    original = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    dest = tmp_path / "weights.safetensors"
    _save_tensors(dest, original)
    assert dest.is_file()
    torch.testing.assert_close(load_file(str(dest))["weight"], original["weight"])


@pytest.fixture
def tiny_gpt2():
    """Testler için 2 katmanlı yapay bir GPT-2 sağlar."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    config = transformers.GPT2Config(
        vocab_size=101, n_positions=32, n_ctx=32, n_embd=32,
        n_layer=2, n_head=4, bos_token_id=0, eos_token_id=1,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    return transformers.GPT2LMHeadModel(config).eval()


def test_cpu_disk_stream_matches_reference(tmp_path, tiny_gpt2):
    """Diskten okumanın referans modelle eşleştiğini ve belleğin temizlendiğini doğrular."""
    output = export_gpt2_shards(tiny_gpt2, tmp_path / "model")
    runtime = DiskOffloadedGPT2(output)

    # Başlangıçta hiçbir katman RAM'de olmamalı (tümü meta olmalı)
    assert runtime.resident_layer_count() == 0
    assert all(p.is_meta for b in runtime.model.transformer.h for p in b.parameters())

    # Farklı batch ve dizi uzunluklarında test ediyoruz
    for ids in ([[1, 5, 9, 7]], [[1, 3], [2, 4]], [[1]]):
        tokens = torch.tensor(ids, dtype=torch.long)
        with torch.inference_mode():
            expected = tiny_gpt2(tokens, use_cache=False).logits
            actual = runtime.logits(tokens)
            # İkinci kez çalıştırıyoruz: Önceki çalıştırmadan RAM'de artık kalmış mı kontrolü
            again = runtime.logits(tokens)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(again, expected, atol=1e-5, rtol=1e-5)
        # İşlem bitince katmanlar yine meta olmalı
        assert runtime.resident_layer_count() == 0
        assert all(p.is_meta for b in runtime.model.transformer.h for p in b.parameters())


def test_layer_file_budget(tmp_path, tiny_gpt2):
    """Katman boyutu bütçeyi (max_layer_file_bytes) aşarsa MemoryError veriyor mu?"""
    output = export_gpt2_shards(tiny_gpt2, tmp_path / "model")
    with pytest.raises(MemoryError, match="bütçeden büyük"):
        DiskOffloadedGPT2(output, max_layer_file_bytes=1)  # 1 bayt limit verip hata bekliyoruz


def test_input_validation(tmp_path, tiny_gpt2):
    """Hatalı girdi şekli ve eksik inference_mode durumunda doğru hataları fırlatıyor mu?"""
    runtime = DiskOffloadedGPT2(export_gpt2_shards(tiny_gpt2, tmp_path / "model"))
    with pytest.raises(ValueError, match="shape"):
        with torch.inference_mode():
            runtime.logits(torch.tensor([1, 2, 3]))
    with pytest.raises(RuntimeError, match="inference_mode"):
        runtime.logits(torch.tensor([[1, 2, 3]]))