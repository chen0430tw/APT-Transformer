#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test HF model loading and generation on cluster"""

from apt.model.hf_compat.modeling_apt import APTForCausalLM
from transformers import AutoTokenizer
import torch

print("[1/3] 加载模型...")
model = APTForCausalLM.from_pretrained("./test_output/hf_converted")
print("  Model type:", type(model).__name__)
print("  Parameters:", sum(p.numel() for p in model.parameters()))

print("\n[2/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./test_output/hf_converted")
print("  Tokenizer OK, vocab_size =", tokenizer.vocab_size)

print("\n[3/3] Testing generation...")
text = "[EN] Hello"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
result = tokenizer.decode(output[0], skip_special_tokens=True)
print("  Input:", text)
print("  Output:", result)

print("\n✅ 所有测试通过！")
