"""
bu modül, dil modelinin mimarisini katman katman manuel olarak yürütmekten ve modelin her bir katmanıın bellek/parametre raporunu cıkarmaktan sorumludur
"""

import torch

def manual_gpt2_logits(model,input_ids:torch.Tensor) -> torch.Tensor:

    if not isinstance(input_ids,torch.Tensor) or input_ids.ndim !=2:
        raise ValueError("input_ids must have shape [batch,squence]/girdi boyutu [batch,squence] tarzında olmalıdır")

    if input_ids.dtype not in (torch.int64,torch.long):
        raise ValueError("input_ids must contain int64 token IDs")

    if input_ids.shape[1]==0:
        raise ValueError("sequence length must be greater than zero")

    if input_ids.shape[1]>model.config.n_positions:
        raise ValueError("Sequence exceeds the model's position capacity")

    if model.training:
        raise ValueError("Model must be in eval() mode")

    transformer = model.transformer
    batch, sequence = input_ids.shape
    positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)

    hidden=transformer.wte(input_ids) + transformer.wpe(positions)
    hidden=transformer.drop(hidden)


    for block in transformer.h:
        result=block(hidden,use_cache=False)
        hidden = result[0] if isinstance(result, (tuple, list)) else result
        if hidden.ndim!=3or hidden.shape[:2] != (batch,sequence):
            raise RuntimeError(f"Unexpected GPT-2 block output shape: {tuple(hidden.shape)}")

    hidden=transformer.ln_f(hidden)
    return model.lm_head(hidden)


def layer_report(model):
    report=[]

    for i,block in enumerate(model.transformer.h):
        parametes=list(block.parameters())
        report.append({
            "layer": i,
            "parameters": sum(p.numel() for p in parametes),
            # Bayt cinsinden hesaplayıp 2^20'ye (1024*1024) bölerek MiB buluyoruz
            "weight_mib": round(sum(p.numel() * p.element_size() for p in parametes) / 2 ** 20, 2),
        })
    return report
