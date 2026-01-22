#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrieval Provider Interface

Defines the interface for RAG (Retrieval-Augmented Generation) implementations.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn

from apt.core.registry import Provider


class RetrievalProvider(Provider):
    """
    Abstract interface for retrieval providers.

    Retrieval providers implement RAG-style retrieval mechanisms for
    augmenting generation with external knowledge.

    Configuration keys (example):
        - retrieval_corpus: Path to retrieval corpus
        - top_k: Number of documents to retrieve
        - embedding_model: Name of embedding model
        - index_type: Type of search index ('faiss', 'annoy', 'exact')
        - fusion_method: How to fuse retrieved context ('concat', 'cross_attn')
    """

    @abstractmethod
    def create_retriever(
        self,
        d_model: int,
        top_k: int = 5,
        **kwargs
    ) -> nn.Module:
        """
        Create a retrieval module.

        Args:
            d_model: Model dimension
            top_k: Number of documents to retrieve
            **kwargs: Additional implementation-specific parameters

        Returns:
            PyTorch retrieval module

        Example:
            provider = registry.get('retrieval', 'faiss_default')
            retriever = provider.create_retriever(d_model=768, top_k=5)
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        retriever: nn.Module,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieve relevant documents given a query.

        Args:
            retriever: Retrieval module instance
            query: Query tensor [batch, seq_len, d_model]
            top_k: Number of documents to retrieve (overrides default)
            **kwargs: Additional parameters

        Returns:
            Tuple of:
            - documents: List of retrieved document strings
            - scores: Relevance scores [batch, top_k]

        Example:
            docs, scores = provider.retrieve(retriever, hidden_states)
        """
        pass

    def build_index(
        self,
        corpus: List[str],
        embedding_model: Optional[nn.Module] = None,
        **kwargs
    ) -> Any:
        """
        Build retrieval index from corpus.

        Args:
            corpus: List of document strings
            embedding_model: Model to encode documents
            **kwargs: Additional parameters

        Returns:
            Index object (implementation-specific)
        """
        raise NotImplementedError(
            "Index building not implemented for this retriever"
        )

    def encode_query(
        self,
        query: torch.Tensor,
        encoder: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Encode query for retrieval.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            encoder: Optional encoder module

        Returns:
            Encoded query [batch, d_query]
        """
        # Default: mean pooling
        return query.mean(dim=1)

    def encode_document(
        self,
        document: str,
        encoder: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Encode document for indexing.

        Args:
            document: Document string
            encoder: Optional encoder module

        Returns:
            Encoded document tensor [d_doc]
        """
        raise NotImplementedError(
            "Document encoding not implemented for this retriever"
        )

    def fuse_context(
        self,
        hidden_states: torch.Tensor,
        retrieved_docs: List[str],
        scores: torch.Tensor,
        method: str = 'concat'
    ) -> torch.Tensor:
        """
        Fuse retrieved context with model hidden states.

        Args:
            hidden_states: Model hidden states [batch, seq_len, d_model]
            retrieved_docs: Retrieved documents
            scores: Relevance scores [batch, top_k]
            method: Fusion method ('concat', 'cross_attn', 'weighted_sum')

        Returns:
            Fused hidden states [batch, seq_len, d_model]
        """
        raise NotImplementedError(
            f"Fusion method '{method}' not implemented"
        )

    def supports_online_retrieval(self) -> bool:
        """
        Check if this retriever supports online retrieval (during training).

        Returns:
            True if online retrieval is supported
        """
        return False

    def supports_batch_retrieval(self) -> bool:
        """
        Check if this retriever supports batch retrieval.

        Returns:
            True if batch retrieval is supported
        """
        return True

    def get_index_size(self) -> int:
        """
        Get the size of the retrieval index (number of documents).

        Returns:
            Number of indexed documents
        """
        return 0

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate retrieval-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        if 'd_model' in config and config['d_model'] <= 0:
            return False

        if 'top_k' in config and config['top_k'] <= 0:
            return False

        return True
