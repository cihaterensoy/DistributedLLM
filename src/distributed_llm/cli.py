"""
Terminal üzerinden çalışan komut satırı arayüzü.
Referans çalıştırma ile manuel katman çalıştırmasını karşılaştırır.
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .engine import manual_gpt2_logits, layer_report

def main():
    #1)komut satırı argümanları

    parser = argparse.ArgumentParser(description="DistributedLLM Phase 0-1")
