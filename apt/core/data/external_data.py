#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
External data loading and processing functionality for APT Model training.
Supports various file formats including TXT, CSV, JSON, JSONL, Excel files.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union, Any, Tuple
from transformers import PreTrainedTokenizer

# Default logger
logger = logging.getLogger('apt_model.data')

def load_external_data(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """
    Load external text dataset from various file formats.
    
    Args:
        file_path (str): Path to the data file
        max_samples (int, optional): Maximum number of samples to load, None for all
        
    Returns:
        list: List of loaded text samples
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported or invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # Select appropriate loading method based on file extension
        if file_extension == '.txt':
            texts = _load_txt_file(file_path, max_samples)
        elif file_extension == '.csv':
            texts = _load_csv_file(file_path, max_samples)
        elif file_extension == '.json':
            texts = _load_json_file(file_path, max_samples)
        elif file_extension == '.jsonl' or file_extension == '.ndjson':
            texts = _load_jsonl_file(file_path, max_samples)
        elif file_extension in ['.xlsx', '.xls']:
            texts = _load_excel_file(file_path, max_samples)
        elif file_extension == '.md':
            texts = _load_md_file(file_path, max_samples)
        else:
            supported_formats = ['.txt', '.csv', '.json', '.jsonl', '.ndjson', '.xlsx', '.xls', '.md']
            raise ValueError(f"Unsupported file format: {file_extension}. "
                           f"Supported formats are: {', '.join(supported_formats)}")
        
        logger.info(f"Successfully loaded {len(texts)} text samples from {file_path}")
        return texts
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def _load_md_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """将 Markdown 文件按段落加载为纯文本列表。"""
    import re
    _MD_PATTERNS = [
        re.compile(r'```[\s\S]*?```'),
        re.compile(r'`[^`]+`'),
        re.compile(r'!\[.*?\]\(.*?\)'),
        re.compile(r'\[([^\]]+)\]\([^\)]+\)'),
        re.compile(r'^#{1,6}\s+', re.M),
        re.compile(r'^\s*[-*+]\s+', re.M),
        re.compile(r'^\s*\d+\.\s+', re.M),
        re.compile(r'\*{1,3}([^*]+)\*{1,3}'),
        re.compile(r'_{1,3}([^_]+)_{1,3}'),
        re.compile(r'^>\s+', re.M),
        re.compile(r'^---+$', re.M),
        re.compile(r'\|.*?\|'),
        re.compile(r'<!--[\s\S]*?-->'),
        re.compile(r'<[^>]+>'),
    ]
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    for pat in _MD_PATTERNS:
        raw = pat.sub(lambda m: m.group(1) if m.lastindex else ' ', raw)
    paragraphs = re.split(r'\n{2,}', raw)
    texts = [p.strip() for p in paragraphs if len(p.strip()) >= 20]
    if max_samples:
        texts = texts[:max_samples]
    return texts


def _load_txt_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load text data from a plain text file, one sample per line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Remove empty lines and strip whitespace
    texts = [line.strip() for line in lines if line.strip()]
    
    # Apply maximum sample limit if specified
    if max_samples and len(texts) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (from {len(texts)} total)")
        texts = texts[:max_samples]
    
    return texts

def _load_csv_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load text data from a CSV file, with interactive column selection."""
    import csv
    
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Read header
        try:
            headers = next(reader)
            print(f"CSV file contains the following columns: {headers}")
        except StopIteration:
            raise ValueError("CSV file appears to be empty")
            
        # Let user select text column
        col_idx = _select_csv_column(headers)
        
        # Read data
        for i, row in enumerate(reader):
            if row and len(row) > col_idx and row[col_idx].strip():
                texts.append(row[col_idx].strip())
            
            # Stop if we've reached the maximum sample count
            if max_samples and len(texts) >= max_samples:
                break
                
    return texts

def _select_csv_column(headers: List[str]) -> int:
    """
    Interactive function to let user select a column from CSV headers.
    
    Args:
        headers: List of column names
        
    Returns:
        int: Selected column index
    """
    while True:
        col_idx_input = input("Please enter the text column index (0-based) or column name: ")
        
        try:
            # Check if input is a number (column index)
            if col_idx_input.isdigit():
                col_idx = int(col_idx_input)
                if col_idx < 0 or col_idx >= len(headers):
                    print(f"Index out of range. Should be between 0 and {len(headers)-1}")
                    continue
            # Otherwise, treat as column name
            else:
                if col_idx_input not in headers:
                    print(f"Column name '{col_idx_input}' not found in {headers}")
                    continue
                col_idx = headers.index(col_idx_input)
            
            return col_idx
            
        except Exception as e:
            print(f"Invalid input: {e}")

def _load_json_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load text data from a JSON file with interactive field selection."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
    
    texts = []
    
    # Handle different JSON structures
    if isinstance(data, list):
        if not data:
            raise ValueError("JSON file contains an empty list")
            
        # If list contains dictionaries, let user select a field
        if isinstance(data[0], dict):
            fields = list(data[0].keys())
            print(f"JSON data contains the following fields: {fields}")
            field = _select_json_field(fields)
            
            for i, item in enumerate(data):
                if field in item and isinstance(item[field], str):
                    texts.append(item[field])
                
                if max_samples and len(texts) >= max_samples:
                    break
        
        # If list contains strings directly
        elif all(isinstance(item, str) for item in data):
            texts = data
            if max_samples:
                texts = texts[:max_samples]
                
        else:
            raise ValueError("Unsupported JSON structure. Expected a list of objects with text fields or a list of strings.")
    
    # Handle case where JSON is a single object
    elif isinstance(data, dict):
        fields = list(data.keys())
        print(f"JSON data contains the following fields: {fields}")
        field = _select_json_field(fields)
        
        # Handle different value types for the selected field
        if isinstance(data[field], list):
            for item in data[field]:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
            
            if max_samples:
                texts = texts[:max_samples]
        
        elif isinstance(data[field], str):
            texts = [data[field]]
    
    else:
        raise ValueError("Unsupported JSON structure")
    
    return texts

def _select_json_field(fields: List[str]) -> str:
    """
    Interactive function to let user select a field from JSON structure.
    
    Args:
        fields: List of available field names
        
    Returns:
        str: Selected field name
    """
    while True:
        field_name = input("Please enter the field name containing text data: ")
        
        if field_name in fields:
            return field_name
        else:
            print(f"Field '{field_name}' not found. Available fields: {fields}")

def _load_jsonl_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load text data from a JSONL (newline-delimited JSON) file."""
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError("JSONL file is empty")
        
    # Parse the first object to determine the structure
    try:
        first_obj = json.loads(lines[0])
    except json.JSONDecodeError:
        raise ValueError("Invalid JSONL format")
    
    # If the object is a dictionary, let user select a field
    if isinstance(first_obj, dict):
        fields = list(first_obj.keys())
        print(f"JSONL data contains the following fields: {fields}")
        field = _select_json_field(fields)
        
        # Process all lines
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                if field in obj and isinstance(obj[field], str):
                    texts.append(obj[field])
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Skipping invalid JSONL line {i+1}")
            
            if max_samples and len(texts) >= max_samples:
                break
    
    # If the object is a string, use it directly
    elif isinstance(first_obj, str):
        texts = [json.loads(line) for line in lines[:max_samples if max_samples else None]]
    
    else:
        raise ValueError("Unsupported JSONL structure")
    
    return texts

def _load_excel_file(file_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load text data from an Excel file with interactive sheet and column selection."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas library is required to load Excel files. "
                         "Install it with: pip install pandas")
    
    # Load Excel file
    excel_data = pd.ExcelFile(file_path)
    
    # Let user select a sheet
    sheet_names = excel_data.sheet_names
    print(f"Excel file contains the following sheets: {sheet_names}")
    
    sheet_name = _select_excel_sheet(sheet_names)
    
    # Load selected sheet
    df = pd.read_excel(excel_data, sheet_name)
    
    # Let user select a column
    columns = df.columns.tolist()
    print(f"Sheet '{sheet_name}' contains the following columns: {columns}")
    
    column = _select_excel_column(columns)
    
    # Extract text data
    texts = df[column].astype(str).tolist()
    
    # Filter out empty strings and NaN values
    texts = [text for text in texts if text and text.lower() != 'nan']
    
    # Apply maximum sample limit if specified
    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]
    
    return texts

def _select_excel_sheet(sheet_names: List[str]) -> str:
    """
    Interactive function to let user select a sheet from Excel file.
    
    Args:
        sheet_names: List of sheet names
        
    Returns:
        str: Selected sheet name
    """
    if len(sheet_names) == 1:
        print(f"Using the only available sheet: {sheet_names[0]}")
        return sheet_names[0]
        
    while True:
        sheet_input = input(f"Please enter the sheet name or index (0-{len(sheet_names)-1}): ")
        
        try:
            # Check if input is a number (sheet index)
            if sheet_input.isdigit():
                sheet_idx = int(sheet_input)
                if sheet_idx < 0 or sheet_idx >= len(sheet_names):
                    print(f"Index out of range. Should be between 0 and {len(sheet_names)-1}")
                    continue
                return sheet_names[sheet_idx]
            
            # Otherwise, treat as sheet name
            elif sheet_input in sheet_names:
                return sheet_input
            else:
                print(f"Sheet '{sheet_input}' not found in {sheet_names}")
        
        except Exception as e:
            print(f"Invalid input: {e}")

def _select_excel_column(columns: List[str]) -> str:
    """
    Interactive function to let user select a column from Excel sheet.
    
    Args:
        columns: List of column names
        
    Returns:
        str: Selected column name
    """
    while True:
        col_input = input(f"Please enter the column name or index (0-{len(columns)-1}): ")
        
        try:
            # Check if input is a number (column index)
            if col_input.isdigit():
                col_idx = int(col_input)
                if col_idx < 0 or col_idx >= len(columns):
                    print(f"Index out of range. Should be between 0 and {len(columns)-1}")
                    continue
                return columns[col_idx]
            
            # Otherwise, treat as column name
            elif col_input in columns:
                return col_input
            else:
                print(f"Column '{col_input}' not found in {columns}")
        
        except Exception as e:
            print(f"Invalid input: {e}")

def preprocess_texts(texts: List[str], min_length: int = 5, max_length: Optional[int] = None) -> List[str]:
    """
    Preprocess text samples with basic cleaning and filtering.
    
    Args:
        texts: List of text samples
        min_length: Minimum text length to keep (in characters)
        max_length: Maximum text length to keep (in characters), None for no limit
        
    Returns:
        list: List of preprocessed text samples
    """
    processed = []
    
    for text in texts:
        # Basic cleaning
        text = text.strip()
        
        # Length filtering
        if len(text) < min_length:
            continue
            
        if max_length and len(text) > max_length:
            text = text[:max_length]
            
        processed.append(text)
    
    return processed

def split_dataset(texts: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1, 
                 test_ratio: float = 0.1, seed: int = 42) -> Dict[str, List[str]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        texts: List of text samples
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with 'train', 'val', and 'test' splits
        
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    import random
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle data
    shuffled = texts.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    train_end = int(len(shuffled) * train_ratio)
    val_end = int(len(shuffled) * (train_ratio + val_ratio))
    
    # Split data
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

def train_with_external_data(data_path=None, epochs=10, batch_size=8,
                            learning_rate=3e-5, save_path="apt_model_custom",
                            max_samples=None, tokenizer=None, language=None,
                            logger=None, custom_texts=None):
    """
    使用外部数据文件或已加载的文本数据训练模型。
    
    参数:
        data_path: 外部数据文件路径，可以为None如果提供了custom_texts
        epochs: 训练轮数
        batch_size: 训练批次大小
        learning_rate: 学习率
        save_path: 保存模型的路径
        max_samples: 使用外部数据的最大样本数
        tokenizer: 预先创建的分词器（可选）
        language: 语言代码（'en'或'zh'，可选）
        logger: 日志记录器
        custom_texts: 已加载的自定义文本数据（可选）
        
    返回:
        tuple: (model, tokenizer, config) 或 (None, None, None)（如果取消）
    """
    import traceback  # 确保导入traceback模块
    
    # 设置本地日志记录器（如果未提供）
    local_logger = logger or logging.getLogger('apt_model.external_training')
    
    try:
        # 处理文本数据
        if custom_texts is not None:
            # 使用已提供的文本数据
            external_texts = custom_texts
            local_logger.info(f"使用已加载的文本数据，共 {len(external_texts)} 条文本")
        elif data_path:
            # 从文件加载文本数据
            local_logger.info(f"从 {data_path} 加载外部数据")
            external_texts = load_external_data(data_path, max_samples)
        else:
            local_logger.error("必须提供data_path或custom_texts中的一个")
            return None, None, None
        
        if not external_texts:
            local_logger.error("未加载有效数据。请检查文件内容。")
            return None, None, None
            
        local_logger.info(f"成功加载 {len(external_texts)} 条文本样本")
        
        # 显示数据样本
        print("\n数据样本:")
        for i in range(min(3, len(external_texts))):
            preview = external_texts[i][:100] + "..." if len(external_texts[i]) > 100 else external_texts[i]
            print(f"[{i+1}] {preview}")
        
        # 只有从文件加载的数据才需要确认训练
        if custom_texts is None:
            confirm = input("\n使用此数据开始训练? (y/n): ")
            if confirm.lower() != 'y':
                local_logger.info("用户取消了训练")
                return None, None, None
        
        # 如果没有提供分词器，使用自动检测
        if not tokenizer:
            from apt.model.tokenization.chinese_tokenizer_integration import get_appropriate_tokenizer
            tokenizer, detected_language = get_appropriate_tokenizer(external_texts, language=language)
            local_logger.info(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
            print(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
            language = detected_language
        
        # 获取基础训练数据作为补充
        try:
            from apt.core.data.base_data import get_training_texts
        except ImportError:
            # 如果找不到特定的base_data模块，尝试从trainer导入
            from apt.trainops.engine.trainer import get_training_texts
            
        base_texts = get_training_texts()
        local_logger.info(f"使用 {len(base_texts)} 条基础训练样本作为补充")
        
        # 合并数据
        train_texts = external_texts + base_texts
        local_logger.info(f"最终训练数据集大小: {len(train_texts)} 条文本样本")
        
        # 开始训练
        from apt.trainops.engine.trainer import train_model
        return train_model(
            texts=train_texts,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path=save_path,
            logger=local_logger,
            tokenizer=tokenizer,
            language=language
        )
        
    except Exception as e:
        local_logger.error(f"使用外部数据训练时出错: {e}")
        local_logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        return None, None, None

def save_dataset(texts: List[str], output_path: str, format: str = 'txt') -> bool:
    """
    Save processed dataset to a file.
    
    Args:
        texts: List of text samples to save
        output_path: Path to save the dataset
        format: Output format ('txt', 'csv', 'json', 'jsonl')
        
    Returns:
        bool: Success status
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(texts))
                
        elif format == 'csv':
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['text'])
                for text in texts:
                    writer.writerow([text])
                    
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)
                
        elif format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
            
        logger.info(f"Dataset saved to {output_path} in {format} format")
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset to {output_path}: {e}")
        return False

if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="External data loading tool")
    parser.add_argument('file_path', help="Path to the data file")
    parser.add_argument('--max-samples', type=int, default=None, 
                      help="Maximum number of samples to load")
    parser.add_argument('--preprocess', action='store_true',
                      help="Apply basic preprocessing")
    parser.add_argument('--min-length', type=int, default=5,
                      help="Minimum text length for preprocessing")
    parser.add_argument('--output', type=str, default=None,
                      help="Output file path for processed data")
    parser.add_argument('--format', type=str, default='txt', 
                      choices=['txt', 'csv', 'json', 'jsonl'],
                      help="Output file format")
    
    args = parser.parse_args()
    
    try:
        texts = load_external_data(args.file_path, args.max_samples)
        print(f"Loaded {len(texts)} text samples")
        
        if args.preprocess:
            texts = preprocess_texts(texts, min_length=args.min_length)
            print(f"After preprocessing: {len(texts)} text samples")
        
        if args.output:
            save_dataset(texts, args.output, args.format)
            print(f"Saved to {args.output} in {args.format} format")
        
        # Print sample texts
        print("\nSample texts:")
        for i in range(min(3, len(texts))):
            preview = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
            print(f"[{i+1}] {preview}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)