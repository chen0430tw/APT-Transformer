#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAISS-based Retrieval Provider

Implements RAG retrieval using FAISS (Facebook AI Similarity Search) for
efficient similarity search in high-dimensional spaces.
"""

import os
import pickle
from typing import Dict, Any, Optional, List, Tuple
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
import numpy as np

from apt.core.providers.retrieval import RetrievalProvider
from apt.core.registry import register


class FaissRetrieverModule(nn.Module):
    """
    FAISS-based retrieval module.

    Supports multiple FAISS index types:
    - 'flat': Exact search (IndexFlatL2)
    - 'ivf': Inverted file index for faster search (IndexIVFFlat)
    - 'hnsw': Hierarchical Navigable Small World graphs (IndexHNSWFlat)
    - 'pq': Product quantization for memory efficiency (IndexPQ)
    """

    def __init__(
        self,
        d_model: int,
        top_k: int = 5,
        index_type: str = 'flat',
        nlist: int = 100,  # For IVF
        nprobe: int = 10,  # For IVF search
        M: int = 32,  # For HNSW
        nbits: int = 8,  # For PQ
        corpus: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.M = M
        self.nbits = nbits
        self.cache_dir = cache_dir or "./cache/faiss"

        # Lazy import FAISS
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu"
            )

        # Initialize index
        self.index = None
        self.corpus = corpus or []
        self.is_trained = False

        # Query encoder (learnable projection)
        self.query_encoder = nn.Linear(d_model, d_model)

        # Context fusion layers
        self.context_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        self.fusion_gate = nn.Linear(d_model * 2, d_model)

    def build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index from pre-computed embeddings.

        Args:
            embeddings: Document embeddings [num_docs, d_model]
        """
        num_docs, d = embeddings.shape
        assert d == self.d_model, f"Embedding dim {d} != d_model {self.d_model}"

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create index based on type
        if self.index_type == 'flat':
            # Exact search using L2 distance
            self.index = self.faiss.IndexFlatL2(self.d_model)
            self.index.add(embeddings)
            self.is_trained = True

        elif self.index_type == 'ivf':
            # Inverted file index for faster search
            quantizer = self.faiss.IndexFlatL2(self.d_model)
            self.index = self.faiss.IndexIVFFlat(
                quantizer, self.d_model, self.nlist
            )
            # Train index
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.nprobe
            self.is_trained = True

        elif self.index_type == 'hnsw':
            # HNSW graph index
            self.index = self.faiss.IndexHNSWFlat(self.d_model, self.M)
            self.index.add(embeddings)
            self.is_trained = True

        elif self.index_type == 'pq':
            # Product quantization for memory efficiency
            m = self.d_model // self.nbits  # Number of subquantizers
            self.index = self.faiss.IndexPQ(self.d_model, m, self.nbits)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.is_trained = True

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        print(f"[FAISS] Built {self.index_type} index with {num_docs} documents")

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
            distances: [batch, top_k]
            indices: [batch, top_k]
        """
        if self.index is None or not self.is_trained:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k or self.top_k

        # Convert to numpy and normalize
        query_np = query_embeddings.detach().cpu().numpy().astype('float32')
        self.faiss.normalize_L2(query_np)

        # Search
        distances, indices = self.index.search(query_np, k)

        # Convert back to torch
        distances = torch.from_numpy(distances).to(query_embeddings.device)
        indices = torch.from_numpy(indices).to(query_embeddings.device)

        return distances, indices

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

        # Convert distances to scores (similarity)
        scores = 1.0 / (1.0 + distances)

        result = {
            'retrieved_indices': indices,
            'retrieval_scores': scores,
        }

        if return_context:
            # TODO: Implement context fusion
            # For now, just return original states
            result['fused_states'] = hidden_states

        return result

    def save_index(self, path: Optional[str] = None):
        """Save FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")

        save_path = path or os.path.join(self.cache_dir, f"faiss_{self.index_type}.index")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.faiss.write_index(self.index, save_path)

        # Save corpus
        corpus_path = save_path.replace('.index', '_corpus.pkl')
        with open(corpus_path, 'wb') as f:
            pickle.dump(self.corpus, f)

        print(f"[FAISS] Saved index to {save_path}")

    def load_index(self, path: Optional[str] = None):
        """Load FAISS index from disk."""
        load_path = path or os.path.join(self.cache_dir, f"faiss_{self.index_type}.index")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Index not found: {load_path}")

        self.index = self.faiss.read_index(load_path)
        self.is_trained = True

        # Load corpus
        corpus_path = load_path.replace('.index', '_corpus.pkl')
        if os.path.exists(corpus_path):
            with open(corpus_path, 'rb') as f:
                self.corpus = pickle.load(f)

        print(f"[FAISS] Loaded index from {load_path}")


@register('retrieval', 'faiss_default')
class FaissRetrievalProvider(RetrievalProvider):
    """
    FAISS-based retrieval provider implementation.

    Configuration:
        index_type: 'flat', 'ivf', 'hnsw', 'pq'
        top_k: Number of documents to retrieve
        nlist: Number of clusters for IVF (default: 100)
        nprobe: Number of clusters to search for IVF (default: 10)
        M: HNSW graph parameter (default: 32)
        nbits: Bits per subquantizer for PQ (default: 8)
        corpus: List of documents
        cache_dir: Directory to cache index
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.index_type = self.config.get('index_type', 'flat')
        self.nlist = self.config.get('nlist', 100)
        self.nprobe = self.config.get('nprobe', 10)
        self.M = self.config.get('M', 32)
        self.nbits = self.config.get('nbits', 8)
        self.corpus = self.config.get('corpus', [])
        self.cache_dir = self.config.get('cache_dir', './cache/faiss')

    def create_retriever(
        self,
        d_model: int,
        top_k: int = 5,
        **kwargs
    ) -> FaissRetrieverModule:
        """Create FAISS retriever module."""
        return FaissRetrieverModule(
            d_model=d_model,
            top_k=top_k,
            index_type=self.index_type,
            nlist=self.nlist,
            nprobe=self.nprobe,
            M=self.M,
            nbits=self.nbits,
            corpus=self.corpus,
            cache_dir=self.cache_dir,
        )

    def retrieve(
        self,
        retriever: FaissRetrieverModule,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieve documents using FAISS.

        Args:
            retriever: FaissRetrieverModule instance
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
    ) -> Any:
        """
        Build FAISS index from corpus.

        Args:
            corpus: List of document strings
            embedding_model: Model to encode documents (uses simple bag-of-words if None)
            **kwargs: Additional parameters

        Returns:
            Embeddings array [num_docs, d_model]
        """
        if embedding_model is None:
            raise ValueError("embedding_model is required for FAISS indexing")

        # Encode all documents
        embeddings_list = []

        with torch.no_grad():
            for doc in corpus:
                # TODO: Implement proper tokenization and encoding
                # For now, this is a placeholder
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
            # Placeholder: use random embedding
            d_model = self.config.get('d_model', 768)
            return torch.randn(d_model)

        with torch.no_grad():
            # TODO: Proper tokenization and encoding
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

        Args:
            hidden_states: [batch, seq_len, d_model]
            retrieved_docs: Retrieved documents
            scores: Relevance scores [batch, top_k]
            method: Fusion method

        Returns:
            Fused states [batch, seq_len, d_model]
        """
        # TODO: Implement proper context fusion
        # For now, just return original states
        return hidden_states

    def supports_online_retrieval(self) -> bool:
        """FAISS supports online retrieval."""
        return True

    def supports_batch_retrieval(self) -> bool:
        """FAISS supports batch retrieval."""
        return True

    def get_index_size(self) -> int:
        """Get corpus size."""
        return len(self.corpus)


# Register variants
@register('retrieval', 'faiss_flat')
class FaissFlatProvider(FaissRetrievalProvider):
    """FAISS with exact flat search."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['index_type'] = 'flat'
        super().__init__(config)


@register('retrieval', 'faiss_ivf')
class FaissIVFProvider(FaissRetrievalProvider):
    """FAISS with IVF (inverted file) index."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['index_type'] = 'ivf'
        super().__init__(config)


@register('retrieval', 'faiss_hnsw')
class FaissHNSWProvider(FaissRetrievalProvider):
    """FAISS with HNSW graph index."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['index_type'] = 'hnsw'
        super().__init__(config)
