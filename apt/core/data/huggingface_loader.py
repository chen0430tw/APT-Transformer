#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hugging Face dataset loader for APT Model.
Provides functionality to load and preprocess datasets from the Hugging Face hub.
"""

import os
import logging
import re
import random
from typing import List, Dict, Union, Optional, Tuple, Any, TYPE_CHECKING

# Import optional dependencies
try:
    from datasets import load_dataset, Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    # Define dummy types for type checking when datasets is not available
    if TYPE_CHECKING:
        from typing import Any as Dataset
        from typing import Any as DatasetDict
    else:
        Dataset = Any
        DatasetDict = Any

class HuggingFaceLoader:
    """
    Loader for Hugging Face datasets.
    Handles loading, column detection, and preprocessing of datasets from the Hugging Face hub.
    """
    
    def __init__(self, logger=None, cache_dir=None):
        """
        Initialize the Hugging Face dataset loader.
        
        Args:
            logger: Optional logger for tracking operations
            cache_dir: Optional directory for caching datasets
        """
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = cache_dir
        
        if not DATASETS_AVAILABLE:
            self.logger.warning(
                "Hugging Face datasets library not available. "
                "Please install with: pip install datasets"
            )
    
    def check_availability(self) -> bool:
        """Check if the datasets library is available."""
        return DATASETS_AVAILABLE
    
    def load_dataset(self, 
                    dataset_name: str, 
                    config_name: Optional[str] = None,
                    split: str = "train", 
                    text_column: Optional[str] = None,
                    max_samples: Optional[int] = None,
                    seed: int = 42) -> Tuple[List[str], Dict[str, Any]]:
        """
        Load a dataset from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset to load
            config_name: Optional configuration name for the dataset
            split: Split to load (train, test, validation)
            text_column: Name of the text column, if None will be auto-detected
            max_samples: Maximum number of samples to load
            seed: Random seed for sampling
        
        Returns:
            Tuple containing (list of text samples, dataset info dictionary)
        """
        if not DATASETS_AVAILABLE:
            self.logger.error("Cannot load dataset: Hugging Face datasets library not available")
            return [], {"error": "datasets library not available"}
        
        self.logger.info(f"Loading dataset: {dataset_name} (config: {config_name}, split: {split})")
        
        try:
            # Load the dataset
            kwargs = {"split": split, "cache_dir": self.cache_dir}
            if config_name:
                kwargs["name"] = config_name
            
            dataset = load_dataset(dataset_name, **kwargs)
            
            # Get dataset info
            info = self._get_dataset_info(dataset)
            
            # Auto-detect text column if not specified
            if text_column is None:
                text_column = self._detect_text_column(dataset)
                self.logger.info(f"Auto-detected text column: '{text_column}'")
                info["detected_text_column"] = text_column
            
            if text_column not in dataset.column_names:
                available_columns = ', '.join(dataset.column_names)
                self.logger.error(f"Column '{text_column}' not found. Available columns: {available_columns}")
                return [], {"error": f"Column not found", "available_columns": dataset.column_names}
            
            # Extract text data
            texts = self._extract_text_data(dataset, text_column)
            
            # Limit samples if requested
            if max_samples is not None and max_samples < len(texts):
                random.seed(seed)
                if max_samples < 100:
                    # For small sample sizes, ensure representative sampling
                    indices = random.sample(range(len(texts)), max_samples)
                    texts = [texts[i] for i in indices]
                else:
                    # For larger samples, a simple slice is efficient
                    texts = texts[:max_samples]
                
                self.logger.info(f"Limited to {max_samples} samples (from {len(dataset)} total)")
                info["sampled"] = True
                info["original_size"] = len(dataset)
            
            info["final_size"] = len(texts)
            return texts, info
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            return [], {"error": str(e)}
    
    def _get_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """
        Extract useful information about the dataset.
        
        Args:
            dataset: Hugging Face dataset object
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            "size": len(dataset),
            "columns": dataset.column_names,
        }
        
        # Extract dataset description and other metadata if available
        if hasattr(dataset, "info") and dataset.info.description:
            info["description"] = dataset.info.description
        
        # Add features info
        if hasattr(dataset, "features"):
            info["features"] = {k: str(v) for k, v in dataset.features.items()}
        
        return info
    
    def _detect_text_column(self, dataset: Dataset) -> str:
        """
        Automatically detect the column containing text data.
        
        Args:
            dataset: Hugging Face dataset
            
        Returns:
            Name of the detected text column
        """
        # Common text column names to check first
        common_text_columns = [
            "text", "content", "sentence", "document", "description",
            "question", "answer", "dialogue", "conversation", "article",
            "prompt", "completion", "query", "response", "summary"
        ]
        
        # Check if any common text column names exist
        for col in common_text_columns:
            if col in dataset.column_names:
                # Verify the column actually contains text
                if self._is_text_column(dataset, col):
                    return col
        
        # If no common name found, analyze each column
        for col in dataset.column_names:
            if self._is_text_column(dataset, col):
                return col
        
        # If no clear text column, default to first string column
        for col in dataset.column_names:
            try:
                if isinstance(dataset[0][col], str):
                    return col
            except (IndexError, KeyError, TypeError):
                continue
        
        # Fallback to the first column
        self.logger.warning("Could not confidently detect a text column, using first column")
        return dataset.column_names[0]
    
    def _is_text_column(self, dataset: Dataset, column: str) -> bool:
        """
        Check if a column contains natural language text.
        
        Args:
            dataset: Hugging Face dataset
            column: Column name to check
            
        Returns:
            True if column likely contains natural text, False otherwise
        """
        # Sample some values to check
        try:
            # Get sample size (minimum of 10 or dataset size)
            sample_size = min(10, len(dataset))
            samples = [dataset[i][column] for i in range(sample_size)]
            
            # Check if samples are strings
            if not all(isinstance(s, str) for s in samples):
                return False
            
            # Filter out None or empty strings
            samples = [s for s in samples if s]
            if not samples:
                return False
            
            # Check average length (natural text typically longer than labels)
            avg_length = sum(len(s.split()) for s in samples) / len(samples)
            if avg_length < 3:  # Very short samples unlikely to be main text
                return False
            
            # Check for sentence structures (periods, capital letters)
            has_sentences = any(re.search(r'[A-Z][^.!?]*[.!?]', s) for s in samples)
            
            return has_sentences
        except Exception:
            return False
    
    def _extract_text_data(self, dataset: Dataset, text_column: str) -> List[str]:
        """
        Extract text data from the dataset.
        
        Args:
            dataset: Hugging Face dataset
            text_column: Name of the column containing text
            
        Returns:
            List of text samples
        """
        texts = []
        
        # Process the dataset in batches to handle large datasets efficiently
        for i in range(0, len(dataset), 1000):
            batch = dataset[i:min(i + 1000, len(dataset))]
            batch_texts = [
                str(item[text_column]) for item in batch 
                if text_column in item and item[text_column]
            ]
            texts.extend(batch_texts)
        
        # Clean and filter texts
        texts = [self._clean_text(t) for t in texts if t and isinstance(t, str)]
        texts = [t for t in texts if t]  # Remove empty texts
        
        return texts
    
    def _clean_text(self, text: str) -> str:
        """
        Clean a text sample.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to string if not already
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def list_available_datasets(self, filter_pattern: Optional[str] = None) -> List[str]:
        """
        List available datasets on Hugging Face matching the filter pattern.
        
        Args:
            filter_pattern: Optional regex pattern to filter datasets
            
        Returns:
            List of matching dataset names
        """
        if not DATASETS_AVAILABLE:
            self.logger.error("Cannot list datasets: Hugging Face datasets library not available")
            return []
            
        try:
            from datasets import list_datasets
            
            # Get all datasets
            all_datasets = list_datasets()
            
            # Filter if pattern provided
            if filter_pattern:
                pattern = re.compile(filter_pattern)
                filtered_datasets = [ds for ds in all_datasets if pattern.search(ds)]
                return filtered_datasets
            
            return all_datasets
            
        except Exception as e:
            self.logger.error(f"Error listing datasets: {str(e)}")
            return []
    
    def preview_dataset(self, 
                       dataset_name: str, 
                       config_name: Optional[str] = None,
                       split: str = "train", 
                       text_column: Optional[str] = None,
                       num_samples: int = 3) -> Dict[str, Any]:
        """
        Preview a dataset with sample entries.
        
        Args:
            dataset_name: Name of the dataset
            config_name: Optional configuration name
            split: Dataset split
            text_column: Text column name, auto-detected if None
            num_samples: Number of samples to preview
            
        Returns:
            Dictionary with dataset preview information
        """
        if not DATASETS_AVAILABLE:
            return {"error": "Hugging Face datasets library not available"}
        
        try:
            # Load the dataset
            kwargs = {"split": split, "cache_dir": self.cache_dir}
            if config_name:
                kwargs["name"] = config_name
                
            dataset = load_dataset(dataset_name, **kwargs)
            
            # Get basic info
            info = self._get_dataset_info(dataset)
            
            # Auto-detect text column if not specified
            if text_column is None:
                text_column = self._detect_text_column(dataset)
                info["detected_text_column"] = text_column
            
            # Get samples
            num_samples = min(num_samples, len(dataset))
            samples = []
            
            for i in range(num_samples):
                entry = dataset[i]
                sample = {
                    "index": i,
                    "columns": {}
                }
                
                # Include all columns from the sample
                for col in dataset.column_names:
                    value = entry[col]
                    
                    # Handle special types or truncate overly long text
                    if isinstance(value, str) and len(value) > 200:
                        sample["columns"][col] = value[:200] + "..."
                    else:
                        try:
                            # Convert to simple type if possible
                            sample["columns"][col] = str(value)
                        except:
                            sample["columns"][col] = f"<non-serializable type: {type(value)}>"
                
                samples.append(sample)
            
            info["samples"] = samples
            return info
            
        except Exception as e:
            return {"error": str(e)}

def load_huggingface_dataset(dataset_name: str, 
                           text_column: Optional[str] = None, 
                           split: str = "train", 
                           max_samples: Optional[int] = None,
                           logger: Optional[logging.Logger] = None,
                           cache_dir: Optional[str] = None) -> List[str]:
    """
    Simplified function to load a Hugging Face dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        text_column: Column name containing text, auto-detected if None
        split: Split to load (train, test, validation)
        max_samples: Maximum number of samples to load
        logger: Optional logger instance
        cache_dir: Optional cache directory
        
    Returns:
        List of text samples
    """
    loader = HuggingFaceLoader(logger=logger, cache_dir=cache_dir)
    
    if not loader.check_availability():
        print("Hugging Face datasets library not available")
        print("Please install with: pip install datasets")
        return []
    
    try:
        print(f"Loading dataset: {dataset_name} (split: {split})")
        
        # Parse dataset name and config
        if '/' in dataset_name:
            # Format like "username/dataset_name"
            config_name = None
        elif ':' in dataset_name:
            # Format like "dataset_name:config"
            dataset_name, config_name = dataset_name.split(':', 1)
        else:
            config_name = None
        
        texts, info = loader.load_dataset(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            text_column=text_column,
            max_samples=max_samples
        )
        
        # Display dataset info
        print(f"Dataset loaded: {info.get('final_size', len(texts))} samples")
        if 'detected_text_column' in info:
            print(f"Using auto-detected text column: '{info['detected_text_column']}'")
        
        # Display sample
        if texts:
            preview = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
            print(f"\nSample text: '{preview}'")
        
        return texts
        
    except Exception as e:
        error_msg = f"Error loading Hugging Face dataset: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return []

if __name__ == "__main__":
    # Example usage when run as a script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python huggingface_loader.py <dataset_name> [text_column] [split] [max_samples]")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    text_column = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("huggingface_loader")
    
    # Load the dataset
    texts = load_huggingface_dataset(
        dataset_name=dataset_name,
        text_column=text_column,
        split=split,
        max_samples=max_samples,
        logger=logger
    )
    
    # Display statistics
    print(f"\nLoaded {len(texts)} samples")
    if texts:
        avg_length = sum(len(t.split()) for t in texts) / len(texts)
        print(f"Average sample length: {avg_length:.1f} words")