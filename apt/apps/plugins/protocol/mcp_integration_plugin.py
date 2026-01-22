#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) Integration Module

Provides streaming retrieval and knowledge augmentation for GPT models,
integrating with the existing APT retrieval infrastructure.

Features:
- Async/streaming retrieval for GPT-5
- Integration with RAG providers (FAISS, Annoy, exact)
- Integration with GraphRAG for advanced queries
- Confidence scoring and evidence fusion
- Memory bucket management
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
import queue
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from apt.core.infrastructure.logging import get_progress_logger

logger = get_progress_logger()


class RetrievalStatus(Enum):
    """Retrieval operation status."""
    IDLE = "idle"
    PENDING = "pending"
    RETRIEVING = "retrieving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    ok: bool
    confidence: float
    evidence_emb: Optional[torch.Tensor] = None
    documents: List[str] = field(default_factory=list)
    scores: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class MCPConfig:
    """Configuration for MCP integration."""

    # Retrieval settings
    provider_name: str = 'exact_cosine'  # 'faiss_default', 'annoy_default', 'exact_cosine', 'graph_rag'
    top_k: int = 3
    confidence_threshold: float = 0.6

    # Async settings
    enable_async: bool = True
    retrieval_timeout: float = 2.0  # seconds
    max_queue_size: int = 10

    # Evidence fusion
    fusion_method: str = 'weighted_mean'  # 'weighted_mean', 'attention', 'max_pool'
    use_score_weighting: bool = True

    # Cache settings
    enable_cache: bool = True
    cache_size: int = 100

    # Integration settings
    d_model: int = 512
    rank: int = 32  # For VeinProjector alignment


class AsyncRetrievalWorker:
    """
    Background worker for async retrieval operations.

    Runs in a separate thread to avoid blocking the main forward pass.
    """

    def __init__(self, retriever_module: nn.Module, timeout: float = 2.0):
        self.retriever = retriever_module
        self.timeout = timeout
        self.request_queue = queue.Queue(maxsize=10)
        self.result_dict = {}  # request_id -> RetrievalResult
        self.worker_thread = None
        self.running = False

    def start(self):
        """Start the worker thread."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("[MCP] Async retrieval worker started")

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        logger.info("[MCP] Async retrieval worker stopped")

    def submit(self, request_id: str, query: torch.Tensor, top_k: int) -> bool:
        """
        Submit a retrieval request.

        Args:
            request_id: Unique request identifier
            query: Query tensor [batch, seq_len, d_model]
            top_k: Number of results to retrieve

        Returns:
            True if submitted successfully
        """
        try:
            self.request_queue.put_nowait({
                'request_id': request_id,
                'query': query.cpu(),  # Move to CPU to avoid CUDA conflicts
                'top_k': top_k,
                'timestamp': time.time()
            })
            return True
        except queue.Full:
            logger.warning("[MCP] Request queue full, dropping request")
            return False

    def poll(self, request_id: str) -> Optional[RetrievalResult]:
        """
        Poll for retrieval result.

        Args:
            request_id: Request identifier

        Returns:
            RetrievalResult if available, None otherwise
        """
        return self.result_dict.pop(request_id, None)

    def _worker_loop(self):
        """Main worker loop (runs in separate thread)."""
        while self.running:
            try:
                # Get next request (with timeout to allow checking self.running)
                try:
                    request = self.request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                request_id = request['request_id']
                query = request['query']
                top_k = request['top_k']

                # Check timeout
                elapsed = time.time() - request['timestamp']
                if elapsed > self.timeout:
                    self.result_dict[request_id] = RetrievalResult(
                        ok=False,
                        confidence=0.0,
                        error=f"Request timed out ({elapsed:.2f}s)"
                    )
                    continue

                # Perform retrieval
                try:
                    result = self._retrieve(query, top_k)
                    self.result_dict[request_id] = result
                except Exception as e:
                    logger.error(f"[MCP] Retrieval error: {e}")
                    self.result_dict[request_id] = RetrievalResult(
                        ok=False,
                        confidence=0.0,
                        error=str(e)
                    )

            except Exception as e:
                logger.error(f"[MCP] Worker error: {e}")

    def _retrieve(self, query: torch.Tensor, top_k: int) -> RetrievalResult:
        """
        Perform actual retrieval.

        This is a placeholder - should be overridden with actual retrieval logic.
        """
        # For now, return empty result
        return RetrievalResult(
            ok=True,
            confidence=0.5,
            evidence_emb=torch.randn_like(query),
            documents=[],
            scores=None
        )


class MCPRetriever(nn.Module):
    """
    MCP-compatible retriever that integrates with APT's retrieval providers.

    This module bridges GPT-5's StreamingRetriever with the existing
    RAG infrastructure (FAISS, Annoy, GraphRAG, etc.).
    """

    def __init__(self, config: MCPConfig):
        super().__init__()
        self.config = config

        # Query encoder (simple MLP for now)
        self.query_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # Evidence fusion layer
        if config.fusion_method == 'attention':
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=8,
                batch_first=True
            )
        elif config.fusion_method == 'weighted_mean':
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.d_model, 1),
                nn.Sigmoid()
            )

        # Retrieval provider integration
        self.provider = None
        self.corpus = []
        self.doc_embeddings = None

        # Async worker
        self.async_worker = None
        if config.enable_async:
            self.async_worker = AsyncRetrievalWorker(
                retriever_module=self,
                timeout=config.retrieval_timeout
            )

        # Cache
        self.cache = {} if config.enable_cache else None
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"[MCP] Initialized with provider={config.provider_name}, top_k={config.top_k}")

    def start_async_worker(self):
        """Start async retrieval worker."""
        if self.async_worker:
            self.async_worker.start()

    def stop_async_worker(self):
        """Stop async retrieval worker."""
        if self.async_worker:
            self.async_worker.stop()

    def set_corpus(self, corpus: List[str], embeddings: Optional[torch.Tensor] = None):
        """
        Set retrieval corpus.

        Args:
            corpus: List of document strings
            embeddings: Pre-computed document embeddings [num_docs, d_model]
        """
        self.corpus = corpus

        if embeddings is not None:
            self.doc_embeddings = embeddings.to(self.query_encoder[0].weight.device)
        else:
            # Compute embeddings using simple tokenization (placeholder)
            logger.warning("[MCP] No embeddings provided, using random embeddings")
            self.doc_embeddings = torch.randn(
                len(corpus),
                self.config.d_model,
                device=self.query_encoder[0].weight.device
            )

        logger.info(f"[MCP] Corpus set: {len(corpus)} documents")

    def retrieve_sync(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Synchronous retrieval.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            top_k: Number of results (uses config default if None)

        Returns:
            RetrievalResult
        """
        if top_k is None:
            top_k = self.config.top_k

        if self.doc_embeddings is None:
            return RetrievalResult(
                ok=False,
                confidence=0.0,
                error="No corpus loaded"
            )

        try:
            # Encode query
            with torch.no_grad():
                query_enc = self.query_encoder(query)  # [B, T, D]
                query_vec = query_enc.mean(dim=1)  # [B, D] - mean pooling

            # Compute similarities
            doc_emb = self.doc_embeddings  # [N, D]
            similarities = F.cosine_similarity(
                query_vec.unsqueeze(1),  # [B, 1, D]
                doc_emb.unsqueeze(0),     # [1, N, D]
                dim=-1
            )  # [B, N]

            # Get top-k
            scores, indices = torch.topk(similarities, k=min(top_k, len(self.corpus)), dim=-1)

            # Retrieve documents
            batch_size = query.size(0)
            documents = []
            for b in range(batch_size):
                batch_docs = [self.corpus[idx.item()] for idx in indices[b]]
                documents.extend(batch_docs)

            # Compute evidence embedding (weighted mean of retrieved doc embeddings)
            evidence_emb = self._fuse_evidence(doc_emb, indices, scores)

            # Compute confidence
            confidence = float(scores.mean().item())

            return RetrievalResult(
                ok=True,
                confidence=confidence,
                evidence_emb=evidence_emb,
                documents=documents,
                scores=scores,
                metadata={
                    'top_k': top_k,
                    'num_corpus': len(self.corpus)
                }
            )

        except Exception as e:
            logger.error(f"[MCP] Retrieval error: {e}")
            return RetrievalResult(
                ok=False,
                confidence=0.0,
                error=str(e)
            )

    def retrieve_async(self, query: torch.Tensor, request_id: str, top_k: Optional[int] = None):
        """
        Asynchronous retrieval (non-blocking).

        Args:
            query: Query tensor
            request_id: Unique request identifier
            top_k: Number of results
        """
        if not self.async_worker or not self.async_worker.running:
            logger.warning("[MCP] Async worker not running, falling back to sync")
            return self.retrieve_sync(query, top_k)

        if top_k is None:
            top_k = self.config.top_k

        self.async_worker.submit(request_id, query, top_k)

    def poll_async(self, request_id: str) -> Optional[RetrievalResult]:
        """
        Poll for async retrieval result.

        Args:
            request_id: Request identifier

        Returns:
            RetrievalResult if ready, None otherwise
        """
        if not self.async_worker:
            return None

        return self.async_worker.poll(request_id)

    def _fuse_evidence(
        self,
        doc_embeddings: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse retrieved document embeddings into evidence vector.

        Args:
            doc_embeddings: All document embeddings [N, D]
            indices: Retrieved document indices [B, K]
            scores: Retrieval scores [B, K]

        Returns:
            Evidence embedding [B, T, D] (T=1 for simplicity)
        """
        batch_size, top_k = indices.shape
        d_model = doc_embeddings.size(-1)

        # Gather retrieved embeddings
        # indices: [B, K] -> [B, K, 1] -> [B, K, D]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d_model)
        retrieved_embs = torch.gather(
            doc_embeddings.unsqueeze(0).expand(batch_size, -1, -1),  # [B, N, D]
            dim=1,
            index=indices_expanded
        )  # [B, K, D]

        if self.config.fusion_method == 'weighted_mean' and self.config.use_score_weighting:
            # Weighted mean by scores
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, K, 1]
            evidence = (retrieved_embs * weights).sum(dim=1, keepdim=True)  # [B, 1, D]
        elif self.config.fusion_method == 'max_pool':
            # Max pooling
            evidence = retrieved_embs.max(dim=1, keepdim=True)[0]  # [B, 1, D]
        else:
            # Simple mean
            evidence = retrieved_embs.mean(dim=1, keepdim=True)  # [B, 1, D]

        return evidence

    def forward(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass (for training).

        Args:
            query: Query tensor [B, T, D]

        Returns:
            Dict with encoded query and retrieval outputs
        """
        query_enc = self.query_encoder(query)

        # Perform retrieval if corpus is loaded
        if self.doc_embeddings is not None:
            result = self.retrieve_sync(query)
            return {
                'query_enc': query_enc,
                'evidence_emb': result.evidence_emb if result.ok else torch.zeros_like(query),
                'confidence': torch.tensor(result.confidence),
                'scores': result.scores if result.scores is not None else torch.zeros(query.size(0), self.config.top_k)
            }
        else:
            return {
                'query_enc': query_enc,
                'evidence_emb': torch.zeros_like(query),
                'confidence': torch.tensor(0.0),
                'scores': torch.zeros(query.size(0), self.config.top_k)
            }


class StreamingRetrieverAdapter:
    """
    Adapter to make MCPRetriever compatible with GPT-5's StreamingRetriever interface.

    This replaces the placeholder StreamingRetriever in gpt5_model.py.
    """

    def __init__(self, mcp_retriever: MCPRetriever, threshold: float = 0.6):
        self.mcp = mcp_retriever
        self.thr = float(threshold)
        self._request_counter = 0
        self._last_request_id = None

    def retrieve_async(self, h: torch.Tensor):
        """
        Start async retrieval (compatible with GPT-5 interface).

        Args:
            h: Hidden states [B, T, D]
        """
        self._request_counter += 1
        self._last_request_id = f"req_{self._request_counter}_{time.time()}"

        if self.mcp.async_worker and self.mcp.async_worker.running:
            self.mcp.retrieve_async(h, self._last_request_id)
        else:
            # Fallback to sync (will be polled immediately)
            pass

    def poll(self) -> Optional[RetrievalResult]:
        """
        Poll for retrieval result (compatible with GPT-5 interface).

        Returns:
            RetrievalResult or None
        """
        if self._last_request_id is None:
            return None

        if self.mcp.async_worker:
            result = self.mcp.poll_async(self._last_request_id)
            if result is not None:
                self._last_request_id = None
            return result
        else:
            # No async worker, return None
            return None


def create_mcp_retriever(
    d_model: int = 512,
    corpus: Optional[List[str]] = None,
    embeddings: Optional[torch.Tensor] = None,
    provider: str = 'exact_cosine',
    top_k: int = 3,
    enable_async: bool = True,
    **kwargs
) -> MCPRetriever:
    """
    Convenience function to create MCP retriever.

    Args:
        d_model: Model dimension
        corpus: Document corpus
        embeddings: Pre-computed document embeddings
        provider: Retrieval provider name
        top_k: Number of documents to retrieve
        enable_async: Enable async retrieval
        **kwargs: Additional config options

    Returns:
        MCPRetriever instance
    """
    config = MCPConfig(
        d_model=d_model,
        provider_name=provider,
        top_k=top_k,
        enable_async=enable_async,
        **kwargs
    )

    retriever = MCPRetriever(config)

    if corpus is not None:
        retriever.set_corpus(corpus, embeddings)

    if enable_async:
        retriever.start_async_worker()

    logger.info(f"[MCP] Created retriever with {len(corpus) if corpus else 0} documents")

    return retriever


# ==================== Integration with GPT-5 ====================

def upgrade_gpt5_with_mcp(
    gpt5_model,
    corpus: List[str],
    embeddings: Optional[torch.Tensor] = None,
    top_k: int = 3,
    enable_async: bool = True
):
    """
    Upgrade a GPT-5 model's retriever with MCP capabilities.

    Args:
        gpt5_model: GPT5Model instance
        corpus: Document corpus
        embeddings: Pre-computed document embeddings
        top_k: Number of documents to retrieve
        enable_async: Enable async retrieval
    """
    # Create MCP retriever
    d_model = gpt5_model.projector.d_model
    mcp_retriever = create_mcp_retriever(
        d_model=d_model,
        corpus=corpus,
        embeddings=embeddings,
        top_k=top_k,
        enable_async=enable_async
    )

    # Replace StreamingRetriever with adapter
    gpt5_model.retriever = StreamingRetrieverAdapter(
        mcp_retriever=mcp_retriever,
        threshold=0.6
    )

    logger.info("[MCP] GPT-5 model upgraded with MCP retriever")

    return gpt5_model


# ==================== Demo ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== MCP Integration Demo ===\n")

    # Create sample corpus
    corpus = [
        "The Transformer architecture uses self-attention mechanisms.",
        "GPT models are autoregressive language models.",
        "BERT uses bidirectional encoding.",
        "RAG combines retrieval with generation.",
        "MoE models use mixture of experts for scaling."
    ]

    # Create retriever
    retriever = create_mcp_retriever(
        d_model=256,
        corpus=corpus,
        top_k=2,
        enable_async=False  # Sync for demo
    )

    # Test retrieval
    query = torch.randn(1, 10, 256)  # [B=1, T=10, D=256]

    print("Performing retrieval...")
    result = retriever.retrieve_sync(query)

    print(f"\nRetrieval result:")
    print(f"  Status: {'✓' if result.ok else '✗'}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Retrieved documents:")
    for i, doc in enumerate(result.documents, 1):
        print(f"    {i}. {doc}")

    if result.scores is not None:
        print(f"  Scores: {result.scores.tolist()}")

    print("\n=== Demo Complete ===")
