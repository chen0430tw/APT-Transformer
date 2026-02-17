#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cache management module for APT Model training system.
Provides functionality for managing, storing, and retrieving cached data files.
"""

import os
import time
import shutil
import logging
import json
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set


class CacheManager:
    """
    Cache manager for APT Model.
    
    Manages various cache files used by the application, including models,
    datasets, tokenizers, checkpoints, logs, and visualizations.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Cache directory path, defaults to ~/.apt_cache if None
            logger: Logger instance for logging cache operations
        """
        self.logger = logger
        
        # Set main cache directory
        if cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.apt_cache")
        else:
            self.cache_dir = os.path.abspath(cache_dir)
        
        # Ensure main cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define cache subdirectories.
        # Note: The "visualizations" directory is changed to "report" folder inside the project.
        # Assuming that this file is in apt_model/utils, then project root is one level up.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_dir = os.path.join(project_root, "report")
        
        self.subdirs = {
            "models": os.path.join(self.cache_dir, "models"),
            "datasets": os.path.join(self.cache_dir, "datasets"),
            "tokenizers": os.path.join(self.cache_dir, "tokenizers"),
            "checkpoints": os.path.join(self.cache_dir, "checkpoints"),
            "logs": os.path.join(self.cache_dir, "logs"),
            "visualizations": report_dir,  # 修改为项目内的 report 目录
            "temp": os.path.join(self.cache_dir, "temp")
        }
        
        # Ensure all subdirectories exist
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        if self.logger:
            self.logger.debug(f"Cache manager initialized, cache directory: {self.cache_dir}")
    
    def clean_cache(self, cache_type: Optional[str] = None, days: int = 30, 
                    exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Clean cache files.
        
        Args:
            cache_type: Type of cache to clean, None for all types
            days: Clean files older than this many days, 0 for all files
            exclude: List of file/directory patterns to exclude from cleaning
            
        Returns:
            dict: Cleaning results with counts and errors
        """
        # Initialize result
        result = {
            'cleaned_files': 0,
            'cleaned_dirs': 0,
            'errors': [],
            'skipped': 0
        }
        
        # Handle exclusion list
        if exclude is None:
            exclude = []
        
        # Determine directories to clean
        dirs_to_clean = []
        if cache_type is None:
            # Clean all cache types
            dirs_to_clean = list(self.subdirs.values())
        elif cache_type in self.subdirs:
            # Clean specific type
            dirs_to_clean = [self.subdirs[cache_type]]
        else:
            if self.logger:
                self.logger.warning(f"Unknown cache type: {cache_type}")
            return result
        
        # Calculate cutoff time (seconds)
        if days > 0:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
        else:
            cutoff_time = float('inf')  # Clean all files
        
        # Start cleaning
        for dir_path in dirs_to_clean:
            if not os.path.exists(dir_path):
                continue
            
            # Clean files
            for root, dirs, files in os.walk(dir_path, topdown=False):
                # Process files
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check if in exclusion list
                    if any(fnmatch.fnmatch(file_path, exc) for exc in exclude):
                        result['skipped'] += 1
                        continue
                    
                    try:
                        # Check file modification time
                        mtime = os.path.getmtime(file_path)
                        if mtime < cutoff_time:
                            # Remove file
                            os.remove(file_path)
                            result['cleaned_files'] += 1
                            if self.logger:
                                self.logger.debug(f"Deleted file: {file_path}")
                    except Exception as e:
                        error_msg = f"Failed to delete file {file_path}: {str(e)}"
                        result['errors'].append(error_msg)
                        if self.logger:
                            self.logger.error(error_msg)
                
                # Process empty directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    
                    # Check if in exclusion list
                    if any(fnmatch.fnmatch(dir_path, exc) for exc in exclude):
                        result['skipped'] += 1
                        continue
                    
                    try:
                        # Check if directory is empty
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            result['cleaned_dirs'] += 1
                            if self.logger:
                                self.logger.debug(f"Deleted empty directory: {dir_path}")
                    except Exception as e:
                        error_msg = f"Failed to delete directory {dir_path}: {str(e)}"
                        result['errors'].append(error_msg)
                        if self.logger:
                            self.logger.error(error_msg)
        
        # Log cleanup results
        if self.logger:
            self.logger.info(f"Cache cleanup completed. Deleted {result['cleaned_files']} files and {result['cleaned_dirs']} directories")
            if result['errors']:
                self.logger.warning(f"Encountered {len(result['errors'])} errors during cleanup")
            if result['skipped']:
                self.logger.info(f"Skipped {result['skipped']} files/directories")
        
        return result

    def get_cache_path(self, subdir_type: str, filename: str) -> str:
        """
        Get cache file path.
        
        Args:
            subdir_type: Subdirectory type
            filename: Filename
        
        Returns:
            str: Full file path
        """
        if subdir_type not in self.subdirs:
            if self.logger:
                self.logger.warning(f"Unknown cache subdirectory type: {subdir_type}, using temp")
            subdir_type = "temp"
        
        # Ensure subdirectory exists
        os.makedirs(self.subdirs[subdir_type], exist_ok=True)
        
        return os.path.join(self.subdirs[subdir_type], filename)

    def get_cache_size(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache size.
        
        Args:
            cache_type: Cache type, None for all types
            
        Returns:
            dict: Cache size information
        """
        result = {
            'total_size_bytes': 0,
            'total_size_human': '',
            'by_type': {}
        }
        
        # Determine directories to check
        dirs_to_check = []
        if cache_type is None:
            # Get all cache types
            dirs_to_check = list(self.subdirs.items())
        elif cache_type in self.subdirs:
            # Get specific type
            dirs_to_check = [(cache_type, self.subdirs[cache_type])]
        else:
            if self.logger:
                self.logger.warning(f"Unknown cache type: {cache_type}")
            return result
        
        # Calculate size for each type
        for type_name, dir_path in dirs_to_check:
            if not os.path.exists(dir_path):
                result['by_type'][type_name] = {'size_bytes': 0, 'size_human': '0B', 'files': 0}
                continue
            
            size = 0
            files = 0
            
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    try:
                        size += os.path.getsize(file_path)
                        files += 1
                    except Exception:
                        pass
            
            # Convert to human-readable format
            size_human = self._format_size(size)
            
            result['by_type'][type_name] = {
                'size_bytes': size,
                'size_human': size_human,
                'files': files
            }
            
            result['total_size_bytes'] += size
        
        # Convert total size to human-readable format
        result['total_size_human'] = self._format_size(result['total_size_bytes'])
        
        return result

    def _format_size(self, size_bytes: int) -> str:
        """
        Convert bytes to human-readable size format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            str: Human-readable size string
        """
        if size_bytes == 0:
            return "0B"
        
        size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"{size_bytes:.2f}{size_names[i]}"

    def save_to_cache(self, data: Union[str, bytes, object], subdir_type: str, 
                    filename: str, overwrite: bool = False) -> Optional[str]:
        """
        Save data to cache.
    
        Args:
            data: Data to save or file path
            subdir_type: Subdirectory type
            filename: Filename
            overwrite: Whether to overwrite existing file
        
        Returns:
            str: Saved file path, None if failed
        """
        try:
            # Get cache file path
            cache_path = self.get_cache_path(subdir_type, filename)
        
            # Check if file already exists
            if os.path.exists(cache_path) and not overwrite:
                if self.logger:
                    self.logger.debug(f"Cache file already exists, not overwriting: {cache_path}")
                return cache_path
        
            # 检查可用存储空间
            cache_dir = os.path.dirname(cache_path)
            try:
                import shutil
                free_space = shutil.disk_usage(cache_dir).free
            
                # 估计数据大小
                if isinstance(data, (str, bytes)):
                    data_size = len(data)
                elif isinstance(data, str) and os.path.isfile(data):
                    data_size = os.path.getsize(data)
                else:
                    # 默认假设至少需要1MB空间
                    data_size = 1024 * 1024
            
                # 检查空间是否足够（额外增加10%作为缓冲）
                if free_space < data_size * 1.1:
                    if self.logger:
                        self.logger.error(f"Insufficient disk space: {self._format_size(free_space)} available, need approximately {self._format_size(data_size)}")
                    return None
            except ImportError:
                # 如果shutil模块不可用，跳过空间检查
                if self.logger:
                    self.logger.warning("Unable to check disk space: shutil module not available")
        
            # 原有的保存逻辑保持不变...
            if isinstance(data, str) and os.path.isfile(data):
                shutil.copy2(data, cache_path)
                if self.logger:
                    self.logger.debug(f"File copied to cache: {cache_path}")
            else:
                # Otherwise, write data to file
                mode = 'wb' if isinstance(data, bytes) else 'w'
                with open(cache_path, mode) as f:
                    f.write(data)
                if self.logger:
                    self.logger.debug(f"Data written to cache: {cache_path}")
        
            return cache_path
    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save to cache: {str(e)}")
            return None

    def load_from_cache(self, subdir_type: str, filename: str, default: Any = None) -> Any:
        """
        Load data from cache.
        
        Args:
            subdir_type: Subdirectory type
            filename: Filename
            default: Default value if file doesn't exist
            
        Returns:
            File content or default value
        """
        try:
            # Get cache file path
            cache_path = self.get_cache_path(subdir_type, filename)
            
            # Check if file exists
            if not os.path.isfile(cache_path):
                if self.logger:
                    self.logger.debug(f"Cache file doesn't exist: {cache_path}")
                return default
            
            # Read and return file content
            with open(cache_path, 'r') as f:
                content = f.read()
            
            if self.logger:
                self.logger.debug(f"Loaded from cache: {cache_path}")
            
            return content
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load from cache: {str(e)}")
            return default

    def load_binary_from_cache(self, subdir_type: str, filename: str, default: Any = None) -> Any:
        """
        Load binary data from cache.
        
        Args:
            subdir_type: Subdirectory type
            filename: Filename
            default: Default value if file doesn't exist
            
        Returns:
            Binary file content or default value
        """
        try:
            # Get cache file path
            cache_path = self.get_cache_path(subdir_type, filename)
            
            # Check if file exists
            if not os.path.isfile(cache_path):
                if self.logger:
                    self.logger.debug(f"Cache file doesn't exist: {cache_path}")
                return default
            
            # Read and return binary file content
            with open(cache_path, 'rb') as f:
                content = f.read()
            
            if self.logger:
                self.logger.debug(f"Loaded binary data from cache: {cache_path}")
            
            return content
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load binary data from cache: {str(e)}")
            return default

    def file_exists_in_cache(self, subdir_type: str, filename: str) -> bool:
        """
        Check if file exists in cache.
        
        Args:
            subdir_type: Subdirectory type
            filename: Filename
            
        Returns:
            bool: Whether file exists
        """
        cache_path = self.get_cache_path(subdir_type, filename)
        return os.path.isfile(cache_path)

    def delete_from_cache(self, subdir_type: str, filename: str) -> bool:
        """
        Delete file from cache.
        
        Args:
            subdir_type: Subdirectory type
            filename: Filename
            
        Returns:
            bool: Whether deletion succeeded
        """
        try:
            cache_path = self.get_cache_path(subdir_type, filename)
            
            if not os.path.isfile(cache_path):
                if self.logger:
                    self.logger.debug(f"Cache file to delete doesn't exist: {cache_path}")
                return False
            
            os.remove(cache_path)
            
            if self.logger:
                self.logger.debug(f"Deleted from cache: {cache_path}")
            
            return True
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to delete from cache: {str(e)}")
            return False

    def list_cache_files(self, subdir_type: Optional[str] = None, 
                         pattern: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        List cache files.
        
        Args:
            subdir_type: Subdirectory type, None for all types
            pattern: Filename pattern (supports wildcards)
            
        Returns:
            dict: Cache file list, grouped by subdirectory
        """
        result = {}
        
        # Determine directories to list
        dirs_to_list = []
        if subdir_type is None:
            # List all cache types
            dirs_to_list = list(self.subdirs.items())
        elif subdir_type in self.subdirs:
            # List specific type
            dirs_to_list = [(subdir_type, self.subdirs[subdir_type])]
        else:
            if self.logger:
                self.logger.warning(f"Unknown cache type: {subdir_type}")
            return result
        
        # Collect files from each directory
        for type_name, dir_path in dirs_to_list:
            if not os.path.exists(dir_path):
                result[type_name] = []
                continue
            
            files = []
            
            for root, _, filenames in os.walk(dir_path):
                rel_root = os.path.relpath(root, dir_path)
                
                for filename in filenames:
                    # If pattern is provided, only include matching files
                    if pattern and not fnmatch.fnmatch(filename, pattern):
                        continue
                    
                    file_path = os.path.join(root, filename)
                    file_size = os.path.getsize(file_path)
                    file_mtime = os.path.getmtime(file_path)
                    
                    # In relative path, "." represents current directory
                    rel_path = filename if rel_root == '.' else os.path.join(rel_root, filename)
                    
                    files.append({
                        'name': filename,
                        'path': rel_path,
                        'full_path': file_path,
                        'size_bytes': file_size,
                        'size_human': self._format_size(file_size),
                        'modified': file_mtime,
                        'modified_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mtime))
                    })
            
            result[type_name] = files
        
        return result

    def get_subdirectory_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about cache subdirectories.
        
        Returns:
            dict: Information about each subdirectory
        """
        result = {}
        
        for type_name, dir_path in self.subdirs.items():
            if not os.path.exists(dir_path):
                result[type_name] = {
                    'path': dir_path,
                    'exists': False,
                    'size_bytes': 0,
                    'size_human': '0B',
                    'files': 0
                }
                continue
            
            # Calculate size and file count
            size = 0
            file_count = 0
            
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    try:
                        size += os.path.getsize(file_path)
                        file_count += 1
                    except Exception:
                        pass
            
            result[type_name] = {
                'path': dir_path,
                'exists': True,
                'size_bytes': size,
                'size_human': self._format_size(size),
                'files': file_count
            }
        
        return result

if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("CacheManager")
    
    parser = argparse.ArgumentParser(description="APT Model Cache Manager")
    parser.add_argument('--cache-dir', type=str, help="Cache directory path")
    parser.add_argument('--action', type=str, required=True, 
                       choices=['info', 'list', 'clean', 'size', 'prune'],
                       help="Action to perform")
    parser.add_argument('--type', type=str, help="Cache type (models, datasets, etc.)")
    parser.add_argument('--days', type=int, default=30, help="Days threshold for pruning")
    parser.add_argument('--pattern', type=str, help="File pattern for listing")
    
    args = parser.parse_args()
    
    # Initialize cache manager
    cm = CacheManager(cache_dir=args.cache_dir, logger=logger)
    
    if args.action == 'info':
        # Show cache info
        info = cm.get_subdirectory_info()
        
        print(f"Cache directory: {cm.cache_dir}")
        print("\nSubdirectories:")
        for type_name, type_info in info.items():
            status = "Exists" if type_info['exists'] else "Not created yet"
            print(f"  {type_name}: {type_info['path']} ({status})")
            if type_info['exists']:
                print(f"    Files: {type_info['files']}, Size: {type_info['size_human']}")
    
    elif args.action == 'list':
        # List cache files
        files = cm.list_cache_files(args.type, args.pattern)
        
        for type_name, type_files in files.items():
            print(f"\n{type_name.upper()} cache files:")
            if not type_files:
                print("  <None>")
                continue
                
            for i, file in enumerate(sorted(type_files, key=lambda x: x['modified'], reverse=True)):
                print(f"  {i+1}. {file['path']} - {file['size_human']} - {file['modified_str']}")
                if i >= 19:  # Show only first 20 files
                    remaining = len(type_files) - 20
                    if remaining > 0:
                        print(f"  ... and {remaining} more files")
                    break
    
    elif args.action == 'clean':
        # Clean all cache
        confirm = input(f"Are you sure you want to clean all files in {cm.cache_dir}? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled")
            sys.exit(0)
            
        result = cm.clean_cache(args.type, days=0)
        print(f"Cleaned {result['cleaned_files']} files and {result['cleaned_dirs']} directories")
        if result['errors']:
            print(f"Encountered {len(result['errors'])} errors")
    
    elif args.action == 'size':
        # Show cache size
        size_info = cm.get_cache_size(args.type)
        
        print(f"Total cache size: {size_info['total_size_human']}")
        print("\nBy type:")
        for type_name, info in size_info['by_type'].items():
            print(f"  {type_name}: {info['size_human']} ({info['files']} files)")
    
    elif args.action == 'prune':
        # Prune old cache files
        print(f"Pruning cache files older than {args.days} days...")
        result = cm.clean_cache(args.type, days=args.days)
        print(f"Cleaned {result['cleaned_files']} files and {result['cleaned_dirs']} directories")
        if result['skipped']:
            print(f"Skipped {result['skipped']} files/directories")
        if result['errors']:
            print(f"Encountered {len(result['errors'])} errors")