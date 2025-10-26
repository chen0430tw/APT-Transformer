#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exact Retrieval Provider

Implements exact brute-force retrieval without approximations.
Simple but accurate baseline for RAG.
"""

import os
import pickle
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from apt.core.providers.retrieval import RetrievalProvider
from apt.core.registry import register


class ExactRetrieverModule(nn.Module):
    """
    Exact brute-force retrieval module.

    Performs exact nearest neighbor search by computing distances to all
    documents. Suitable for small to medium-sized corpora.

    Supports:
    - Cosine similarity
    - L2 (Euclidean) distance
    - Dot product similarity
    """

    def __init__(
        self,
        d_model: int,
        top_k: int = 5,
        metric: str = 'cosine',  # 'cosine', 'l2', 'dot'
        corpus: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.metric = metric
        self.cache_dir = cache_dir or "./cache/exact"
        self.corpus = corpus or []

        # Document embeddings [num_docs, d_model]
        self.doc_embeddings = None

        # Query encoder (learnable projection)
        self.query_encoder = nn.Linear(d_model, d_model)

        # Context fusion layers
        self.context_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def set_doc_embeddings(self, embeddings: torch.Tensor):
        """
        Set document embeddings.

        Args:
            embeddings: [num_docs, d_model]
        """
        assert embeddings.shape[1] == self.d_model
        self.doc_embeddings = embeddings

        # Normalize for cosine similarity
        if self.metric == 'cosine':
            self.doc_embeddings = F.normalize(self.doc_embeddings, p=2, dim=1)

    def compute_similarity(
        self,
        query: torch.Tensor,
        docs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between queries and documents.

        Args:
            query: [batch, d_model]
            docs: [num_docs, d_model]

        Returns:
            scores: [batch, num_docs]
        """
        if self.metric == 'cosine':
            # Normalize query
            query = F.normalize(query, p=2, dim=1)
            # Cosine similarity = dot product of normalized vectors
            scores = torch.matmul(query, docs.t())  # [batch, num_docs]

        elif self.metric == 'dot':
            # Dot product
            scores = torch.matmul(query, docs.t())  # [batch, num_docs]

        elif self.metric == 'l2':
            # L2 distance (convert to similarity)
            # Expand dimensions for broadcasting
            query_exp = query.unsqueeze(1)  # [batch, 1, d_model]
            docs_exp = docs.unsqueeze(0)  # [1, num_docs, d_model]
            # Compute L2 distance
            distances = torch.norm(query_exp - docs_exp, p=2, dim=2)  # [batch, num_docs]
            # Convert to similarity (higher is better)
            scores = -distances

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return scores

    def search(
        self,
        query_embeddings: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for nearest neighbors.

        Args:
            query_embeddings: Query embeddings [batch, d_model]
            top_k: Number of results (default: self.top_k)

        Returns:
            scores: [batch, top_k]
            indices: [batch, top_k]
        """
        if self.doc_embeddings is None:
            raise RuntimeError("Document embeddings not set. Call set_doc_embeddings() first.")

        k = top_k or self.top_k
        k = min(k, len(self.doc_embeddings))

        # Compute similarity to all documents
        scores = self.compute_similarity(query_embeddings, self.doc_embeddings)  # [batch, num_docs]

        # Get top-k
        top_scores, indices = torch.topk(scores, k, dim=1)  # [batch, top_k]

        return top_scores, indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_context: bool = True,
        fusion_method: str = 'gate'
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve and optionally fuse context.

        Args:
            hidden_states: [batch, seq_len, d_model]
            return_context: If True, return fused context
            fusion_method: 'gate', 'attention', or 'concat'

        Returns:
            Dict with keys:
                - retrieved_indices: [batch, top_k]
                - retrieval_scores: [batch, top_k]
                - fused_states: [batch, seq_len, d_model] (if return_context)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Encode query (mean pooling + projection)
        query = hidden_states.mean(dim=1)  # [batch, d_model]
        query = self.query_encoder(query)  # [batch, d_model]

        # Search
        scores, indices = self.search(query)  # [batch, top_k]

        result = {
            'retrieved_indices': indices,
            'retrieval_scores': scores,
        }

        if return_context:
            # Fuse retrieved context with hidden states
            fused = self._fuse_context(
                hidden_states, indices, scores, method=fusion_method
            )
            result['fused_states'] = fused

        return result

    def _fuse_context(
        self,
        hidden_states: torch.Tensor,
        doc_indices: torch.Tensor,
        doc_scores: torch.Tensor,
        method: str = 'gate'
    ) -> torch.Tensor:
        """
        Fuse retrieved document embeddings with hidden states.

        Args:
            hidden_states: [batch, seq_len, d_model]
            doc_indices: [batch, top_k]
            doc_scores: [batch, top_k]
            method: Fusion method

        Returns:
            Fused states [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        top_k = doc_indices.shape[1]

        # Get retrieved document embeddings
        retrieved_embeds = self.doc_embeddings[doc_indices]  # [batch, top_k, d_model]

        # Normalize scores to weights
        weights = F.softmax(doc_scores, dim=1)  # [batch, top_k]

        # Weighted sum of retrieved embeddings
        weighted_context = torch.bmm(
            weights.unsqueeze(1),  # [batch, 1, top_k]
            retrieved_embeds  # [batch, top_k, d_model]
        ).squeeze(1)  # [batch, d_model]

        # Expand to sequence length
        context_exp = weighted_context.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]

        if method == 'gate':
            # Gated fusion
            combined = torch.cat([hidden_states, context_exp], dim=-1)  # [batch, seq_len, 2*d_model]
            gate = self.fusion_gate(combined)  # [batch, seq_len, d_model]
            fused = gate * hidden_states + (1 - gate) * context_exp

        elif method == 'attention':
            # Cross-attention fusion
            # Use retrieved embeddings as keys/values
            retrieved_exp = retrieved_embeds.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, seq_len, top_k, d_model]
            retrieved_flat = retrieved_exp.reshape(batch_size * seq_len, top_k, d_model)  # [batch*seq_len, top_k, d_model]
            hidden_flat = hidden_states.reshape(batch_size * seq_len, 1, d_model)  # [batch*seq_len, 1, d_model]

            # Cross-attention
            attn_out, _ = self.context_attention(
                hidden_flat, retrieved_flat, retrieved_flat
            )  # [batch*seq_len, 1, d_model]

            fused = attn_out.reshape(batch_size, seq_len, d_model)

        elif method == 'concat':
            # Simple concatenation + projection
            combined = torch.cat([hidden_states, context_exp], dim=-1)  # [batch, seq_len, 2*d_model]
            fused = self.fusion_proj(combined)  # [batch, seq_len, d_model]

        else:
            raise ValueError(f"Unknown fusion method: {method}")

        return fused

    def save_index(self, path: Optional[str] = None):
        """Save document embeddings and corpus to disk."""
        if self.doc_embeddings is None:
            raise RuntimeError("No embeddings to save")

        save_path = path or os.path.join(self.cache_dir, f"exact_{self.metric}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_data = {
            'doc_embeddings': self.doc_embeddings.cpu(),
            'corpus': self.corpus,
            'd_model': self.d_model,
            'metric': self.metric,
        }
        torch.save(save_data, save_path)

        print(f"[Exact] Saved index to {save_path}")

    def load_index(self, path: Optional[str] = None, device: str = 'cpu'):
        """Load document embeddings and corpus from disk."""
        load_path = path or os.path.join(self.cache_dir, f"exact_{self.metric}.pt")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Index not found: {load_path}")

        save_data = torch.load(load_path, map_location=device)

        self.doc_embeddings = save_data['doc_embeddings'].to(device)
        self.corpus = save_data['corpus']
        self.d_model = save_data.get('d_model', self.d_model)
        self.metric = save_data.get('metric', self.metric)

        print(f"[Exact] Loaded index from {load_path}")


@register('retrieval', 'exact_default')
class ExactRetrievalProvider(RetrievalProvider):
    """
    Exact brute-force retrieval provider implementation.

    Configuration:
        metric: 'cosine', 'l2', 'dot'
        top_k: Number of documents to retrieve
        corpus: List of documents
        cache_dir: Directory to cache embeddings
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.metric = self.config.get('metric', 'cosine')
        self.corpus = self.config.get('corpus', [])
        self.cache_dir = self.config.get('cache_dir', './cache/exact')

    def create_retriever(
        self,
        d_model: int,
        top_k: int = 5,
        **kwargs
    ) -> ExactRetrieverModule:
        """Create exact retriever module."""
        return ExactRetrieverModule(
            d_model=d_model,
            top_k=top_k,
            metric=self.metric,
            corpus=self.corpus,
            cache_dir=self.cache_dir,
        )

    def retrieve(
        self,
        retriever: ExactRetrieverModule,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieve documents using exact search.

        Args:
            retriever: ExactRetrieverModule instance
            query: Query tensor [batch, seq_len, d_model]
            top_k: Number of documents to retrieve

        Returns:
            documents: List of retrieved documents
            scores: Relevance scores [batch, top_k]
        """
        # Get retrieval results
        result = retriever(query, return_context=False)
        indices = result['retrieved_indices']  # [batch, top_k]
        scores = result['retrieval_scores']  # [batch, top_k]

        # Get documents from corpus
        batch_size, k = indices.shape
        documents = []

        for batch_idx in range(batch_size):
            batch_docs = []
            for doc_idx in indices[batch_idx]:
                doc_idx = doc_idx.item()
                if 0 <= doc_idx < len(retriever.corpus):
                    batch_docs.append(retriever.corpus[doc_idx])
                else:
                    batch_docs.append("")
            documents.append(batch_docs)

        # Flatten if single batch
        if batch_size == 1:
            documents = documents[0]

        return documents, scores

    def build_index(
        self,
        corpus: List[str],
        embedding_model: Optional[nn.Module] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Build exact retrieval index from corpus.

        Args:
            corpus: List of document strings
            embedding_model: Model to encode documents
            **kwargs: Additional parameters

        Returns:
            Embeddings tensor [num_docs, d_model]
        """
        if embedding_model is None:
            raise ValueError("embedding_model is required for exact retrieval")

        # Encode all documents
        embeddings_list = []

        with torch.no_grad():
            for doc in corpus:
                # TODO: Implement proper tokenization and encoding
                embedding = torch.randn(self.config.get('d_model', 768))
                embeddings_list.append(embedding)

        embeddings = torch.stack(embeddings_list)

        return embeddings

    def encode_document(
        self,
        document: str,
        encoder: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Encode a single document."""
        if encoder is None:
            d_model = self.config.get('d_model', 768)
            return torch.randn(d_model)

        with torch.no_grad():
            return encoder(document)

    def fuse_context(
        self,
        hidden_states: torch.Tensor,
        retrieved_docs: List[str],
        scores: torch.Tensor,
        method: str = 'concat'
    ) -> torch.Tensor:
        """
        Fuse retrieved context with hidden states.

        The exact retriever implements proper fusion in its forward() method.
        """
        # This is handled by ExactRetrieverModule._fuse_context()
        return hidden_states

    def supports_online_retrieval(self) -> bool:
        """Exact retrieval supports online retrieval."""
        return True

    def supports_batch_retrieval(self) -> bool:
        """Exact retrieval supports batch retrieval."""
        return True

    def get_index_size(self) -> int:
        """Get corpus size."""
        return len(self.corpus)


# Register metric-specific variants
@register('retrieval', 'exact_cosine')
class ExactCosineProvider(ExactRetrievalProvider):
    """Exact retrieval with cosine similarity."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['metric'] = 'cosine'
        super().__init__(config)


@register('retrieval', 'exact_l2')
class ExactL2Provider(ExactRetrievalProvider):
    """Exact retrieval with L2 distance."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['metric'] = 'l2'
        super().__init__(config)


@register('retrieval', 'exact_dot')
class ExactDotProvider(ExactRetrievalProvider):
    """Exact retrieval with dot product similarity."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['metric'] = 'dot'
        super().__init__(config)
