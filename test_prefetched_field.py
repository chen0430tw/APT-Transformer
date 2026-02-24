#!/usr/bin/env python3
"""测试packed对象的prefetched字段是否能跨线程访问"""

import torch
import torch.autograd
import threading
import time

class Packed:
    def __init__(self):
        self.prefetched = None

pack_hook_calls = 0
unpack_hook_calls = 0

def pack_hook(tensor):
    global pack_hook_calls
    pack_hook_calls += 1
    print(f"[Pack #{pack_hook_calls}] tensor id={id(tensor)}, shape={tensor.shape}")

    # 创建packed对象
    packed = Packed()

    # 模拟：在后台线程中设置prefetched
    def set_prefetched():
        time.sleep(0.01)  # 模拟异步延迟
        packed.prefetched = tensor.detach().cpu()
        print(f"[Pack #{pack_hook_calls}] Set prefetched in background thread, id={id(packed)}, prefetched_id={id(packed.prefetched)}")

    thread = threading.Thread(target=set_prefetched)
    thread.start()

    return packed

def unpack_hook(packed):
    global unpack_hook_calls
    unpack_hook_calls += 1
    print(f"[Unpack #{unpack_hook_calls}] packed id={id(packed)}, prefetched={packed.prefetched is not None}")

    if packed.prefetched is not None:
        print(f"[Unpack #{unpack_hook_calls}] ✅ SUCCESS: prefetched tensor shape={packed.prefetched.shape}")
    else:
        print(f"[Unpack #{unpack_hook_calls}] ❌ FAILED: prefetched is None")

    # 模拟处理延迟
    time.sleep(0.02)

    return packed.prefetched if packed.prefetched is not None else torch.zeros(1)

print("Testing packed.prefetched field visibility across threads...")
print("=" * 70)

a = torch.ones(5, requires_grad=True)
b = torch.ones(5, requires_grad=True)

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = a * b

print("\n" + "=" * 70)
print("Waiting for threads to complete...")
time.sleep(0.1)

print(f"\nTotal: {pack_hook_calls} pack calls, {unpack_hook_calls} unpack calls")
print("\nTriggering backward...")
y.sum().backward()
print(f"After backward: {unpack_hook_calls} unpack calls")
