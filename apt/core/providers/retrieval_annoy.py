#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Annoy-based Retrieval Provider

Implements RAG retrieval using Annoy (Approximate Nearest Neighbors Oh Yeah)
for memory-efficient approximate nearest neighbor search.
"""

import os
import pickle
from typing import Dict, Any, Optional, List, Tuple
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
import numpy as np

from apt.core.providers.retrieval import RetrievalProvider
from apt.core.registry import register


class AnnoyRetrieverModule(nn.Module):
    """
    Annoy-based retrieval module.

    Annoy is a C++ library with Python bindings for approximate nearest neighbor
    search, optimized for memory usage and read-only query performance.

    Supports distance metrics:
    - 'angular': Cosine similarity (default)
    - 'euclidean': L2 distance
    - 'manhattan': L1 distance
    - 'hamming': Hamming distance
    - 'dot': Dot product
    """

    def __init__(
        self,
        d_model: int,
        top_k: int = 5,
        metric: str = 'angular',
        n_trees: int = 10,  # More trees = better accuracy but slower build
        search_k: int = -1,  # -1 means n_trees * n, higher = better accuracy
        corpus: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.metric = metric
        self.n_trees = n_trees
        self.search_k = search_k
        self.cache_dir = cache_dir or "./cache/annoy"

        # Lazy import Annoy
        try:
            from annoy import AnnoyIndex
            self.AnnoyIndex = AnnoyIndex
        except ImportError:
            raise ImportError(
                "Annoy not installed. Install with: pip install annoy"
            )

        # Initialize index
        self.index = None
        self.corpus = corpus or []
        self.is_built = False

        # Query encoder (learnable projection)
        self.query_encoder = nn.Linear(d_model, d_model)

    def build_index(self, embeddings: np.ndarray):
        """
        Build Annoy index from pre-computed embeddings.

        Args:
            embeddings: Document embeddings [num_docs, d_model]
        """
        num_docs, d = embeddings.shape
        assert d == self.d_model, f"Embedding dim {d} != d_model {self.d_model}"

        # Create index
        self.index = self.AnnoyIndex(self.d_model, self.metric)

        # Add all embeddings
        for i in range(num_docs):
            self.index.add_item(i, embeddings[i])

        # Build index with n_trees
        self.index.build(self.n_trees)
        self.is_built = True

        print(f"[Annoy] Built index with {num_docs} documents, {self.n_trees} trees, metric={self.metric}")

    def search(
        self,
        query_embeddings: torch.Tensor,
        top_k: Optional[int] = None,
        search_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for nearest neighbors.

        Args:
            query_embeddings: Query embeddings [batch, d_model]
            top_k: Number of results (default: self.top_k)
            search_k: Search parameter (-1 for default)

        Returns:
            distances: [batch, top_k]
            indices: [batch, top_k]
        """
        if self.index is None or not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k or self.top_k
        sk = search_k if search_k is not None else self.search_k

        # Convert to numpy
        query_np = query_embeddings.detach().cpu().numpy().astype('float32')

        batch_size = query_np.shape[0]
        all_indices = []
        all_distances = []

        # Search each query
        for i in range(batch_size):
            indices, distances = self.index.get_nns_by_vector(
                query_np[i],
                k,
                search_k=sk,
                include_distances=True
            )
            all_indices.append(indices)
            all_distances.append(distances)

        # Convert to torch tensors
        indices_tensor = torch.tensor(all_indices, dtype=torch.long).to(query_embeddings.device)
        distances_tensor = torch.tensor(all_distances, dtype=torch.float32).to(query_embeddings.device)

        return distances_tensor, indices_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_context: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve and optionally fuse context.

        Args:
            hidden_states: [batch, seq_len, d_model]
            return_context: If True, return fused context

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
        distances, indices = self.search(query)  # [batch, top_k]

        # Convert distances to scores
        # For angular distance: similarity = 1 - (angular_distance / pi)
        if self.metric == 'angular':
            scores = 1.0 - (distances / np.pi)
        else:
            # For other metrics: use inverse distance
            scores = 1.0 / (1.0 + distances)

        result = {
            'retrieved_indices': indices,
            'retrieval_scores': scores,
        }

        if return_context:
            # TODO: Implement context fusion
            result['fused_states'] = hidden_states

        return result

    def save_index(self, path: Optional[str] = None):
        """Save Annoy index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")

        save_path = path or os.path.join(self.cache_dir, f"annoy_{self.metric}.ann")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.index.save(save_path)

        # Save corpus and metadata
        meta_path = save_path.replace('.ann', '_meta.pkl')
        metadata = {
            'corpus': self.corpus,
            'd_model': self.d_model,
            'metric': self.metric,
            'n_trees': self.n_trees,
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"[Annoy] Saved index to {save_path}")

    def load_index(self, path: Optional[str] = None):
        """Load Annoy index from disk."""
        load_path = path or os.path.join(self.cache_dir, f"annoy_{self.metric}.ann")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Index not found: {load_path}")

        # Load metadata first
        meta_path = load_path.replace('.ann', '_meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
                self.corpus = metadata.get('corpus', [])
                self.d_model = metadata.get('d_model', self.d_model)
                self.metric = metadata.get('metric', self.metric)
                self.n_trees = metadata.get('n_trees', self.n_trees)

        # Load index
        self.index = self.AnnoyIndex(self.d_model, self.metric)
        self.index.load(load_path)
        self.is_built = True

        print(f"[Annoy] Loaded index from {load_path}")

    def get_item_vector(self, idx: int) -> np.ndarray:
        """Get embedding vector for a specific document."""
        if self.index is None:
            raise RuntimeError("Index not loaded")
        return self.index.get_item_vector(idx)

    def get_distance(self, idx1: int, idx2: int) -> float:
        """Get distance between two documents."""
        if self.index is None:
            raise RuntimeError("Index not loaded")
        return self.index.get_distance(idx1, idx2)


@register('retrieval', 'annoy_default')
class AnnoyRetrievalProvider(RetrievalProvider):
    """
    Annoy-based retrieval provider implementation.

    Configuration:
        metric: 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'
        top_k: Number of documents to retrieve
        n_trees: Number of trees to build (default: 10)
        search_k: Search parameter (default: -1)
        corpus: List of documents
        cache_dir: Directory to cache index
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.metric = self.config.get('metric', 'angular')
        self.n_trees = self.config.get('n_trees', 10)
        self.search_k = self.config.get('search_k', -1)
        self.corpus = self.config.get('corpus', [])
        self.cache_dir = self.config.get('cache_dir', './cache/annoy')

    def create_retriever(
        self,
        d_model: int,
        top_k: int = 5,
        **kwargs
    ) -> AnnoyRetrieverModule:
        """Create Annoy retriever module."""
        return AnnoyRetrieverModule(
            d_model=d_model,
            top_k=top_k,
            metric=self.metric,
            n_trees=self.n_trees,
            search_k=self.search_k,
            corpus=self.corpus,
            cache_dir=self.cache_dir,
        )

    def retrieve(
        self,
        retriever: AnnoyRetrieverModule,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieve documents using Annoy.

        Args:
            retriever: AnnoyRetrieverModule instance
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
    ) -> np.ndarray:
        """
        Build Annoy index from corpus.

        Args:
            corpus: List of document strings
            embedding_model: Model to encode documents
            **kwargs: Additional parameters

        Returns:
            Embeddings array [num_docs, d_model]
        """
        if embedding_model is None:
            raise ValueError("embedding_model is required for Annoy indexing")

        # Encode all documents
        embeddings_list = []

        with torch.no_grad():
            for doc in corpus:
                # TODO: Implement proper tokenization and encoding
                embedding = torch.randn(self.config.get('d_model', 768))
                embeddings_list.append(embedding)

        embeddings = torch.stack(embeddings_list).numpy()

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
        """Fuse retrieved context with hidden states."""
        # TODO: Implement proper context fusion
        return hidden_states

    def supports_online_retrieval(self) -> bool:
        """Annoy supports online retrieval (read-only)."""
        return True

    def supports_batch_retrieval(self) -> bool:
        """Annoy supports batch retrieval."""
        return True

    def get_index_size(self) -> int:
        """Get corpus size."""
        return len(self.corpus)


# Register metric-specific variants
@register('retrieval', 'annoy_angular')
class AnnoyAngularProvider(AnnoyRetrievalProvider):
    """Annoy with angular (cosine) distance."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['metric'] = 'angular'
        super().__init__(config)


@register('retrieval', 'annoy_euclidean')
class AnnoyEuclideanProvider(AnnoyRetrievalProvider):
    """Annoy with Euclidean distance."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['metric'] = 'euclidean'
        super().__init__(config)


@register('retrieval', 'annoy_dot')
class AnnoyDotProvider(AnnoyRetrievalProvider):
    """Annoy with dot product similarity."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['metric'] = 'dot'
        super().__init__(config)
