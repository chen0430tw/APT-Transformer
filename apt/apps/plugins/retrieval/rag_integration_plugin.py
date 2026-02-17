#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) Integration Module

Provides high-level utilities for integrating retrieval into APT models.
"""

import os
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from dataclasses import dataclass

from apt.core.registry import registry
from apt.core.system import get_device
from apt.core.infrastructure.logging import get_progress_logger

logger = get_progress_logger()


@dataclass
class RAGConfig:
    """Configuration for RAG integration."""

    # Retrieval provider config
    provider_name: str = 'exact_cosine'  # 'faiss_default', 'annoy_default', 'exact_cosine'
    top_k: int = 5  # Number of documents to retrieve

    # Index config
    index_type: str = 'flat'  # For FAISS: 'flat', 'ivf', 'hnsw', 'pq'
    metric: str = 'cosine'  # 'cosine', 'l2', 'dot', 'angular'

    # Corpus config
    corpus_path: Optional[str] = None
    corpus_encoding: str = 'utf-8'
    max_corpus_size: Optional[int] = None

    # Embedding config
    d_model: int = 768
    use_pretrained_encoder: bool = False
    encoder_name: Optional[str] = None  # e.g., 'sentence-transformers/all-MiniLM-L6-v2'

    # Fusion config
    fusion_method: str = 'gate'  # 'gate', 'attention', 'concat'
    fusion_layer_indices: Optional[List[int]] = None  # Which layers to apply RAG

    # Training config
    train_retriever: bool = True  # Whether to train query encoder
    freeze_index: bool = True  # Whether to freeze document embeddings
    rag_loss_weight: float = 0.1  # Weight for retrieval-related auxiliary loss

    # Cache config
    cache_dir: str = './cache/rag'
    cache_index: bool = True
    load_index_if_exists: bool = True


class RAGWrapper(nn.Module):
    """
    Wraps a language model with RAG capabilities.

    This module adds retrieval-augmented generation to any transformer-based
    language model by injecting retrieved context at specified layers.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: RAGConfig,
        corpus: Optional[List[str]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = get_device()

        # Get retrieval provider from registry
        try:
            provider_class = registry.get('retrieval', config.provider_name)
            self.provider = provider_class({
                'metric': config.metric,
                'index_type': config.index_type,
                'top_k': config.top_k,
                'corpus': corpus or [],
                'cache_dir': config.cache_dir,
                'd_model': config.d_model,
            })
        except Exception as e:
            logger.warning(f"Failed to get retrieval provider '{config.provider_name}': {e}")
            logger.info("Falling back to exact_cosine provider")
            provider_class = registry.get('retrieval', 'exact_cosine')
            self.provider = provider_class({
                'metric': 'cosine',
                'top_k': config.top_k,
                'corpus': corpus or [],
                'cache_dir': config.cache_dir,
                'd_model': config.d_model,
            })

        # Create retriever module
        self.retriever = self.provider.create_retriever(
            d_model=config.d_model,
            top_k=config.top_k
        )

        # Move to device
        self.retriever = self.retriever.to(self.device)

        # Track if index is built
        self.index_built = False

        # Determine which layers to apply RAG
        if config.fusion_layer_indices is None:
            # Default: apply at middle layers
            num_layers = getattr(base_model, 'num_layers', 12)
            self.fusion_layer_indices = [num_layers // 2, num_layers * 3 // 4]
        else:
            self.fusion_layer_indices = config.fusion_layer_indices

        logger.info(f"[RAG] Initialized with provider={config.provider_name}, top_k={config.top_k}")
        logger.info(f"[RAG] Fusion at layers: {self.fusion_layer_indices}")

    def build_index(
        self,
        corpus: Optional[List[str]] = None,
        doc_embeddings: Optional[torch.Tensor] = None,
        embedding_model: Optional[nn.Module] = None
    ):
        """
        Build retrieval index.

        Args:
            corpus: List of documents (will update retriever.corpus)
            doc_embeddings: Pre-computed document embeddings [num_docs, d_model]
            embedding_model: Model to encode documents (if doc_embeddings not provided)
        """
        if corpus is not None:
            self.retriever.corpus = corpus

        if doc_embeddings is None:
            if embedding_model is None:
                logger.warning("[RAG] No embedding model provided, using random embeddings")
                embeddings = self.provider.build_index(
                    self.retriever.corpus,
                    embedding_model=None
                )
            else:
                logger.info(f"[RAG] Encoding {len(self.retriever.corpus)} documents...")
                embeddings = self.provider.build_index(
                    self.retriever.corpus,
                    embedding_model=embedding_model
                )
        else:
            embeddings = doc_embeddings

        # Build index based on retriever type
        if hasattr(self.retriever, 'build_index'):
            # FAISS or Annoy
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            self.retriever.build_index(embeddings)
        elif hasattr(self.retriever, 'set_doc_embeddings'):
            # Exact retriever
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.from_numpy(embeddings)
            embeddings = embeddings.to(self.device)
            self.retriever.set_doc_embeddings(embeddings)
        else:
            raise RuntimeError("Retriever does not support index building")

        self.index_built = True
        logger.info(f"[RAG] Index built with {len(self.retriever.corpus)} documents")

        # Cache index if configured
        if self.config.cache_index:
            self.save_index()

    def load_index(self, path: Optional[str] = None):
        """Load pre-built index from disk."""
        self.retriever.load_index(path)
        self.index_built = True
        logger.info(f"[RAG] Loaded index from {path or 'default path'}")

    def save_index(self, path: Optional[str] = None):
        """Save index to disk."""
        if not self.index_built:
            raise RuntimeError("Index not built yet")
        self.retriever.save_index(path)
        logger.info(f"[RAG] Saved index to {path or 'default path'}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with RAG.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            **kwargs: Additional arguments for base model

        Returns:
            Dict with:
                - logits: [batch, seq_len, vocab_size]
                - retrieved_docs: List of retrieved documents
                - retrieval_scores: [batch, top_k]
        """
        if not self.index_built:
            logger.warning("[RAG] Index not built, running without retrieval")
            outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
            return {
                'logits': outputs.logits if hasattr(outputs, 'logits') else outputs,
                'retrieved_docs': [],
                'retrieval_scores': None,
            }

        # Get base model hidden states
        # We need to hook into intermediate layers for RAG fusion
        hidden_states = None
        retrieved_docs = None
        retrieval_scores = None

        # Simple forward pass (for now)
        # TODO: Implement layer-wise injection of retrieved context
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # Get hidden states from middle layer for retrieval
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            layer_idx = len(outputs.hidden_states) // 2
            hidden_states = outputs.hidden_states[layer_idx]

            # Retrieve documents
            retrieved_docs, retrieval_scores = self.provider.retrieve(
                self.retriever,
                hidden_states,
                top_k=self.config.top_k
            )

        return {
            'logits': outputs.logits if hasattr(outputs, 'logits') else outputs,
            'retrieved_docs': retrieved_docs or [],
            'retrieval_scores': retrieval_scores,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        }

    def retrieve(
        self,
        query: Union[torch.Tensor, str],
        top_k: Optional[int] = None
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Standalone retrieval without generation.

        Args:
            query: Query tensor [batch, seq_len, d_model] or string
            top_k: Number of documents to retrieve

        Returns:
            documents: Retrieved documents
            scores: Relevance scores
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        if isinstance(query, str):
            # TODO: Encode string query
            raise NotImplementedError("String query encoding not implemented yet")

        docs, scores = self.provider.retrieve(
            self.retriever,
            query,
            top_k=top_k
        )

        return docs, scores


def load_corpus_from_file(
    filepath: str,
    encoding: str = 'utf-8',
    max_size: Optional[int] = None
) -> List[str]:
    """
    Load corpus from text file.

    Args:
        filepath: Path to corpus file (one document per line)
        encoding: File encoding
        max_size: Maximum number of documents to load

    Returns:
        List of documents
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    corpus = []
    with open(filepath, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            if max_size and i >= max_size:
                break
            line = line.strip()
            if line:
                corpus.append(line)

    logger.info(f"[RAG] Loaded {len(corpus)} documents from {filepath}")
    return corpus


def create_rag_model(
    base_model: nn.Module,
    config: RAGConfig,
    corpus: Optional[Union[List[str], str]] = None,
    auto_build_index: bool = True
) -> RAGWrapper:
    """
    Create a RAG-enabled model.

    Args:
        base_model: Base language model
        config: RAG configuration
        corpus: Corpus (list of strings or path to file)
        auto_build_index: Whether to automatically build index

    Returns:
        RAG-wrapped model
    """
    # Load corpus if path provided
    if isinstance(corpus, str):
        corpus = load_corpus_from_file(
            corpus,
            encoding=config.corpus_encoding,
            max_size=config.max_corpus_size
        )

    # Create RAG wrapper
    rag_model = RAGWrapper(base_model, config, corpus)

    # Try to load existing index
    if config.load_index_if_exists:
        index_path = os.path.join(
            config.cache_dir,
            f"{config.provider_name.replace('_', '-')}.index"
        )
        if os.path.exists(index_path) or os.path.exists(index_path.replace('.index', '.pt')):
            try:
                rag_model.load_index()
                logger.info("[RAG] Loaded existing index")
                auto_build_index = False
            except Exception as e:
                logger.warning(f"[RAG] Failed to load index: {e}")

    # Build index if needed
    if auto_build_index and corpus:
        rag_model.build_index(corpus=corpus)

    return rag_model


# Convenience function
def quick_rag(
    model: nn.Module,
    corpus: Union[List[str], str],
    provider: str = 'exact_cosine',
    top_k: int = 5,
    **kwargs
) -> RAGWrapper:
    """
    Quick setup for RAG with sensible defaults.

    Args:
        model: Base language model
        corpus: List of documents or path to corpus file
        provider: Retrieval provider name
        top_k: Number of documents to retrieve
        **kwargs: Additional config options

    Returns:
        RAG-enabled model
    """
    config = RAGConfig(
        provider_name=provider,
        top_k=top_k,
        **kwargs
    )

    return create_rag_model(model, config, corpus, auto_build_index=True)
