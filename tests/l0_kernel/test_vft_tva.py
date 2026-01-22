#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for VFT/TVA module
"""

import torch
from apt_model.modeling.blocks import (
    VeinProjector,
    TVAAttention,
    VFTFeedForward,
    VFTBlock,
    create_vft_block,
    get_attention,
    get_ffn,
    list_attention,
    list_ffn,
)


def test_vein_projector():
    """Test VeinProjector."""
    print("\n=== Testing VeinProjector ===")

    d_model, rank = 256, 16
    batch, seq_len = 2, 32

    # Test linear implementation
    vein = VeinProjector(d_model, rank, implementation='linear', init_method='orthogonal')
    x = torch.randn(batch, seq_len, d_model)

    z = vein.project(x)
    x_rec = vein.reconstruct(z)

    print(f"Input shape: {x.shape}")
    print(f"Projected shape: {z.shape}")
    print(f"Reconstructed shape: {x_rec.shape}")

    # Compute reconstruction error
    error = vein.compute_reconstruction_error(x)
    print(f"Reconstruction error shape: {error.shape}")
    print(f"Mean reconstruction error: {error.mean().item():.6f}")
    print(f"Compression ratio: {vein.get_compression_ratio():.1f}x")

    # Test parameter implementation
    vein_param = VeinProjector(d_model, rank, implementation='parameter')
    z2 = vein_param.project(x)
    x_rec2 = vein_param.reconstruct(z2)
    print(f"\nParameter impl - Projected shape: {z2.shape}")

    print("✓ VeinProjector test passed")


def test_tva_attention():
    """Test TVAAttention."""
    print("\n=== Testing TVAAttention ===")

    d_model, n_heads, rank = 256, 8, 16
    batch, seq_len = 2, 32

    attn = TVAAttention(d_model, n_heads, rank, attn_dropout=0.1)
    x = torch.randn(batch, seq_len, d_model)

    # Forward pass
    output, attn_weights = attn(x, return_attention_weights=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    # Test with mask
    mask = torch.zeros(batch, 1, seq_len, seq_len)
    mask[:, :, :, seq_len//2:] = float('-inf')  # Causal mask
    output_masked, _ = attn(x, attn_mask=mask)
    print(f"Output with mask shape: {output_masked.shape}")

    print("✓ TVAAttention test passed")


def test_vft_ffn():
    """Test VFTFeedForward."""
    print("\n=== Testing VFTFeedForward ===")

    d_model, rank = 256, 16
    batch, seq_len = 2, 32

    ffn = VFTFeedForward(d_model, rank, r_hidden=64, activation='silu', dropout=0.1)
    x = torch.randn(batch, seq_len, d_model)

    output = ffn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Check different activations
    for act in ['gelu', 'relu', 'silu']:
        ffn_act = VFTFeedForward(d_model, rank, activation=act)
        out = ffn_act(x)
        print(f"Activation {act}: output shape {out.shape}")

    print("✓ VFTFeedForward test passed")


def test_vft_block():
    """Test VFTBlock."""
    print("\n=== Testing VFTBlock ===")

    d_model, n_heads, rank = 256, 8, 16
    batch, seq_len = 2, 32

    block = VFTBlock(
        d_model=d_model,
        n_heads=n_heads,
        rank=rank,
        s_normals=1,
        tau=0.18,
        attn_dropout=0.0,
        ffn_dropout=0.0,
    )

    x = torch.randn(batch, seq_len, d_model)

    # Forward pass
    output, metrics = block(x, return_metrics=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Metrics: {metrics}")

    # Test with factory
    block2 = create_vft_block(d_model, n_heads, rank)
    output2, _ = block2(x, return_metrics=False)
    print(f"Factory block output shape: {output2.shape}")

    print("✓ VFTBlock test passed")


def test_registry():
    """Test registry system."""
    print("\n=== Testing Registry ===")

    # List available implementations
    print(f"Registered attention: {list_attention()}")
    print(f"Registered FFN: {list_ffn()}")

    # Get implementations via registry
    attn = get_attention('tva', d_model=256, n_heads=8, rank=16)
    ffn = get_ffn('vft', d_model=256, rank=16)

    print(f"TVA from registry: {type(attn).__name__}")
    print(f"VFT from registry: {type(ffn).__name__}")

    # Get standard implementations
    std_attn = get_attention('standard', d_model=256, n_heads=8)
    std_ffn = get_ffn('standard', d_model=256)

    print(f"Standard attention: {type(std_attn).__name__}")
    print(f"Standard FFN: {type(std_ffn).__name__}")

    # Test forward pass
    x = torch.randn(2, 32, 256)
    attn_out, _ = attn(x)
    ffn_out = ffn(x)

    print(f"TVA output shape: {attn_out.shape}")
    print(f"VFT output shape: {ffn_out.shape}")

    print("✓ Registry test passed")


def test_parameter_count():
    """Compare parameter counts."""
    print("\n=== Parameter Count Comparison ===")

    d_model = 768
    n_heads = 12
    rank = 4
    d_ff = 4 * d_model

    # VFT/TVA
    vft_block = VFTBlock(d_model, n_heads, rank)
    vft_params = sum(p.numel() for p in vft_block.parameters())

    # Standard (approximate)
    # Attention: 4 * d_model * d_model (Q, K, V, O)
    # FFN: 2 * d_model * d_ff
    std_attn_params = 4 * d_model * d_model
    std_ffn_params = 2 * d_model * d_ff
    std_total = std_attn_params + std_ffn_params

    print(f"VFT/TVA block parameters: {vft_params:,}")
    print(f"Standard block parameters (approx): {std_total:,}")
    print(f"Reduction: {std_total / vft_params:.2f}x")

    print("✓ Parameter count test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("VFT/TVA Module Test Suite")
    print("=" * 60)

    torch.manual_seed(42)

    test_vein_projector()
    test_tva_attention()
    test_vft_ffn()
    test_vft_block()
    test_registry()
    test_parameter_count()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
