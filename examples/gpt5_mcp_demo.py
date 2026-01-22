#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-5 with MCP Integration Demo

Demonstrates how to use GPT-5 with Model Context Protocol (MCP)
for retrieval-augmented generation.
"""

import torch
import torch.nn as nn
from typing import List

from apt_model.modeling.gpt5_model import GPT5Model
from apt_model.modeling.mcp_integration import (
    create_mcp_retriever,
    upgrade_gpt5_with_mcp,
    MCPConfig
)


def demo_basic_retrieval():
    """Demo 1: Basic MCP retrieval"""
    print("=" * 70)
    print("Demo 1: Basic MCP Retrieval")
    print("=" * 70)

    # Sample corpus
    corpus = [
        "The Transformer architecture revolutionized NLP in 2017.",
        "Self-attention allows models to weigh input tokens differently.",
        "GPT models use autoregressive language modeling.",
        "BERT uses masked language modeling for pretraining.",
        "Mixture of Experts (MoE) enables model scaling.",
        "Retrieval-Augmented Generation combines retrieval with generation.",
        "Knowledge graphs store structured information as triples.",
    ]

    # Create retriever
    retriever = create_mcp_retriever(
        d_model=256,
        corpus=corpus,
        top_k=3,
        enable_async=False  # Sync for demo simplicity
    )

    # Simulate query
    print("\nQuery: [Random embedding simulating 'What is Transformer?']")
    query = torch.randn(1, 10, 256)

    # Retrieve
    result = retriever.retrieve_sync(query)

    print(f"\n✓ Retrieval Status: {'Success' if result.ok else 'Failed'}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"\nTop-{len(result.documents)} Retrieved Documents:")
    for i, doc in enumerate(result.documents, 1):
        score = result.scores[0, i-1].item() if result.scores is not None else 0.0
        print(f"  {i}. [{score:.3f}] {doc}")

    print("\n" + "=" * 70)


def demo_gpt5_integration():
    """Demo 2: GPT-5 with MCP integration"""
    print("\n" + "=" * 70)
    print("Demo 2: GPT-5 + MCP Integration")
    print("=" * 70)

    # Create GPT-5 model
    print("\n1. Creating GPT-5 model...")
    model = GPT5Model(
        vocab_size=1000,  # Small vocab for demo
        d_model=256,
        n_layers=2,
        num_skills=8,
        top_k=2,
        rank=16
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare knowledge corpus
    corpus = [
        "Neural networks consist of interconnected layers.",
        "Backpropagation updates weights using gradients.",
        "Activation functions introduce non-linearity.",
        "Dropout prevents overfitting during training.",
        "Batch normalization stabilizes training.",
    ]
    print(f"\n2. Knowledge corpus: {len(corpus)} documents")

    # Upgrade with MCP
    print("\n3. Upgrading with MCP...")
    model = upgrade_gpt5_with_mcp(
        model,
        corpus=corpus,
        top_k=2,
        enable_async=False
    )
    print("   ✓ MCP integration complete")

    # Test forward pass
    print("\n4. Testing forward pass with retrieval...")
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 16))  # [B=2, T=16]

    with torch.no_grad():
        logits, info = model.forward_step(
            input_ids,
            step_idx=0,
            schema_required=False
        )

    print(f"   Input shape: {tuple(input_ids.shape)}")
    print(f"   Output shape: {tuple(logits.shape)}")
    print(f"   Feedback: {info['feedback']}")
    print(f"   Memory length: {info['mem_len']}")

    if info.get('align'):
        print(f"   Alignment: {info['align']}")

    print("\n" + "=" * 70)


def demo_training_loop():
    """Demo 3: Training with MCP"""
    print("\n" + "=" * 70)
    print("Demo 3: Training Loop with MCP")
    print("=" * 70)

    # Setup
    model = GPT5Model(vocab_size=500, d_model=128, n_layers=2, num_skills=4)

    corpus = [
        "Machine learning learns from data.",
        "Deep learning uses neural networks.",
        "Supervised learning uses labeled data.",
    ]

    model = upgrade_gpt5_with_mcp(model, corpus=corpus, top_k=1, enable_async=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining for 5 steps...")
    model.train()

    for step in range(5):
        # Random batch
        input_ids = torch.randint(0, 500, (2, 20))
        labels = torch.randint(0, 500, (2, 20))

        # Forward (with retrieval)
        logits, info = model.forward_step(input_ids, step_idx=step)

        # Compute loss
        loss = criterion(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"  Step {step+1}: Loss={loss.item():.4f}, "
              f"Entropy={info['feedback']['entropy']:.3f}, "
              f"ΔKL={info['feedback']['dkl']:.4f}")

    print("\n✓ Training complete")
    print("=" * 70)


def demo_custom_config():
    """Demo 4: Custom MCP configuration"""
    print("\n" + "=" * 70)
    print("Demo 4: Custom MCP Configuration")
    print("=" * 70)

    # Create custom config
    config = MCPConfig(
        provider_name='exact_cosine',
        top_k=5,
        confidence_threshold=0.7,
        enable_async=True,
        retrieval_timeout=1.0,
        fusion_method='weighted_mean',
        use_score_weighting=True,
        enable_cache=True,
        cache_size=50,
        d_model=256,
        rank=32
    )

    print("\nCustom Configuration:")
    print(f"  Provider: {config.provider_name}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print(f"  Async: {config.enable_async}")
    print(f"  Timeout: {config.retrieval_timeout}s")
    print(f"  Fusion: {config.fusion_method}")
    print(f"  Cache: {config.enable_cache} (size={config.cache_size})")

    # Create retriever with custom config
    from apt_model.modeling.mcp_integration import MCPRetriever

    corpus = ["Doc 1", "Doc 2", "Doc 3"]
    retriever = MCPRetriever(config)
    retriever.set_corpus(corpus)

    print("\n✓ Retriever created with custom config")
    print("=" * 70)


def demo_async_retrieval():
    """Demo 5: Async retrieval"""
    print("\n" + "=" * 70)
    print("Demo 5: Async Retrieval")
    print("=" * 70)

    import time

    corpus = [f"Document {i} with some content." for i in range(10)]

    # Create retriever with async enabled
    retriever = create_mcp_retriever(
        d_model=128,
        corpus=corpus,
        top_k=3,
        enable_async=True
    )

    print("\n1. Starting async worker...")
    retriever.start_async_worker()
    print("   ✓ Worker started")

    # Submit request
    print("\n2. Submitting async retrieval request...")
    request_id = "demo_request_001"
    query = torch.randn(1, 8, 128)
    retriever.retrieve_async(query, request_id)
    print(f"   ✓ Request '{request_id}' submitted")

    # Do other work
    print("\n3. Doing other work while retrieval happens...")
    time.sleep(0.1)
    print("   ✓ Work done")

    # Poll for result
    print("\n4. Polling for result...")
    result = retriever.poll_async(request_id)

    if result:
        print(f"   ✓ Result ready!")
        print(f"   Status: {'Success' if result.ok else 'Failed'}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Documents: {len(result.documents)}")
    else:
        print("   Result not ready yet (try polling again)")

    # Cleanup
    print("\n5. Stopping async worker...")
    retriever.stop_async_worker()
    print("   ✓ Worker stopped")

    print("\n" + "=" * 70)


def demo_evidence_fusion():
    """Demo 6: Evidence fusion methods"""
    print("\n" + "=" * 70)
    print("Demo 6: Evidence Fusion Methods")
    print("=" * 70)

    corpus = ["Doc A", "Doc B", "Doc C"]
    query = torch.randn(1, 5, 256)

    fusion_methods = ['weighted_mean', 'max_pool']

    for method in fusion_methods:
        print(f"\nTesting fusion method: {method}")

        retriever = create_mcp_retriever(
            d_model=256,
            corpus=corpus,
            top_k=2,
            enable_async=False,
            fusion_method=method
        )

        result = retriever.retrieve_sync(query)

        if result.ok and result.evidence_emb is not None:
            print(f"  ✓ Evidence shape: {tuple(result.evidence_emb.shape)}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Mean activation: {result.evidence_emb.mean().item():.4f}")
        else:
            print(f"  ✗ Failed: {result.error}")

    print("\n" + "=" * 70)


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("GPT-5 + MCP Integration Demo Suite")
    print("=" * 70)

    torch.manual_seed(42)  # For reproducibility

    try:
        # Run demos
        demo_basic_retrieval()
        demo_gpt5_integration()
        demo_training_loop()
        demo_custom_config()
        demo_async_retrieval()
        demo_evidence_fusion()

        print("\n" + "=" * 70)
        print("✓ All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
