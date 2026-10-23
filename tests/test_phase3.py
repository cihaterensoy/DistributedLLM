"""Ağ ve transformers paketi olmadan çalışan streaming entegrasyon testleri."""
import json
import sys
import types
from pathlib import Path
import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from src.distributed_llm.adapters import get_adapter
from src.distributed_llm.weight_index import WeightIndex
from src.distributed_llm.streaming import StreamingCausalLM


class TinyConfig:
    model_type = 'gpt2'
    tie_word_embeddings = True


class TinyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(16, 8)
        self.h = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.ln_f = nn.LayerNorm(8)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TinyBody()
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids, use_cache=False):
        x = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            x = torch.tanh(block(x))
        return types.SimpleNamespace(logits=self.lm_head(self.transformer.ln_f(x)))


def make_model_dir(tmp_path, sharded=False):
    torch.manual_seed(123)
    model = TinyModel().eval()
    weights = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()
               if k != 'lm_head.weight'}
    if sharded:
        keys = sorted(weights)
        halves = [keys[::2], keys[1::2]]
        weight_map = {}
        for idx, names in enumerate(halves):
            name = f'model-{idx + 1:05d}-of-00002.safetensors'
            save_file({k: weights[k] for k in names}, str(tmp_path / name))
            weight_map.update({k: name for k in names})
        (tmp_path / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': weight_map}), encoding='utf-8')
    else:
        save_file(weights, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'config.json').write_text('{"model_type": "gpt2"}', encoding='utf-8')
    return model


@pytest.fixture
def fake_transformers(monkeypatch):
    fake = types.ModuleType('transformers')
    fake.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyConfig())
    fake.AutoModelForCausalLM = types.SimpleNamespace(from_config=lambda *args, **kwargs: TinyModel())
    monkeypatch.setitem(sys.modules, 'transformers', fake)


@pytest.mark.parametrize('sharded', [False, True])
def test_streaming_matches_full_model(tmp_path, fake_transformers, sharded):
    reference = make_model_dir(tmp_path, sharded=sharded)
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with torch.inference_mode():
        expected = reference(inputs).logits
    engine = StreamingCausalLM(tmp_path)
    assert engine.resident_layer_count() == 0
    assert engine.model.lm_head.weight is engine.model.transformer.wte.weight
    result = engine.logits(inputs)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    assert engine.layer_reads == 3
    assert engine.resident_layer_count() == 0
    engine.logits(inputs)
    assert engine.layer_reads == 6
    engine.close()


def test_budget_fail_and_cleanup(tmp_path, fake_transformers):
    make_model_dir(tmp_path)
    engine = StreamingCausalLM(tmp_path, max_layer_bytes=1)
    with pytest.raises(MemoryError):
        engine.logits(torch.tensor([[1, 2]]))
    assert engine.resident_layer_count() == 0


def test_shape_validation(tmp_path, fake_transformers):
    make_model_dir(tmp_path)
    engine = StreamingCausalLM(tmp_path)
    with pytest.raises(ValueError, match='shape'):
        engine.logits(torch.tensor([1, 2]))


def test_weight_index_rejects_path_traversal(tmp_path):
    (tmp_path / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'x': '../private.safetensors'}}), encoding='utf-8')
    with pytest.raises(ValueError):
        WeightIndex(tmp_path)


def test_adapter_registry():
    assert get_adapter('gpt2').prefix(2) == 'transformer.h.2'
    assert get_adapter('llama').prefix(2) == 'model.layers.2'
    assert get_adapter('qwen2').prefix(2) == 'model.layers.2'
    with pytest.raises(ValueError):
        get_adapter('unsupported')
