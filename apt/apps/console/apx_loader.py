#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APX Package Loader

Loads APX packages and automatically configures plugins based on capabilities.
"""

import zipfile
import tempfile
import shutil
import yaml
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class APXLoader:
    """APX package loader with plugin auto-configuration"""

    def __init__(self, extract_dir: Optional[Path] = None):
        """
        Initialize APX loader.

        Args:
            extract_dir: Directory to extract APX packages to.
                        If None, uses temporary directory.
        """
        self.extract_dir = extract_dir
        self._temp_dirs = []  # Track temporary directories for cleanup

    def load(self, apx_path: Path) -> Dict[str, Any]:
        """
        Load APX package.

        Args:
            apx_path: Path to APX file

        Returns:
            Dictionary with:
            - manifest: apx.yaml content
            - artifacts_dir: Extracted artifacts directory
            - adapters_dir: Extracted adapters directory
            - capabilities: Detected capabilities (from manifest or detection)
            - extract_dir: Extraction root directory

        Raises:
            FileNotFoundError: If APX file not found
            ValueError: If APX package is invalid
        """
        if not apx_path.exists():
            raise FileNotFoundError(f"APX file not found: {apx_path}")

        # Determine extraction directory
        if self.extract_dir is None:
            extract_root = Path(tempfile.mkdtemp(prefix="apx_"))
            self._temp_dirs.append(extract_root)
        else:
            extract_root = self.extract_dir / apx_path.stem
            extract_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting APX to: {extract_root}")

        # Extract APX package (ZIP format)
        with zipfile.ZipFile(apx_path, 'r') as zf:
            zf.extractall(extract_root)

        # Read manifest
        manifest_path = extract_root / "apx.yaml"
        if not manifest_path.exists():
            raise ValueError(f"Invalid APX package: apx.yaml not found")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)

        # Get capabilities from manifest
        capabilities = manifest.get("capabilities", {}).get("provides", [])

        # Additional detection from artifacts (if available)
        artifacts_dir = extract_root / "artifacts"
        if artifacts_dir.exists():
            try:
                # Import detector dynamically to avoid circular dependency
                from apt.apps.tools.apx.detectors import detect_capabilities

                detected = detect_capabilities(artifacts_dir)
                # Merge with manifest capabilities
                capabilities = list(set(capabilities + detected))

                logger.info(f"Capabilities detected: {detected}")
            except Exception as e:
                logger.warning(f"Could not auto-detect capabilities: {e}")

        logger.info(f"APX capabilities: {capabilities}")

        return {
            "manifest": manifest,
            "artifacts_dir": artifacts_dir,
            "adapters_dir": extract_root / "model" / "adapters",
            "capabilities": capabilities,
            "extract_dir": extract_root,
        }

    def load_with_auto_plugins(
        self,
        apx_path: Path,
        auto_loader: 'AutoPluginLoader',
        auto_enable: bool = True,
        score_threshold: float = 0.0,
    ) -> Tuple[Dict[str, Any], List['PluginBase']]:
        """
        Load APX package and automatically configure plugins.

        Args:
            apx_path: Path to APX file
            auto_loader: AutoPluginLoader instance
            auto_enable: Whether to auto-enable recommended plugins
            score_threshold: Minimum plugin score to load

        Returns:
            (apx_info, loaded_plugins)
        """
        # Load APX package
        apx_info = self.load(apx_path)

        # Auto-load plugins based on capabilities
        plugins = auto_loader.load_for_capabilities(
            capabilities=apx_info['capabilities'],
            auto_enable=auto_enable,
            score_threshold=score_threshold,
        )

        logger.info(f"Auto-loaded {len(plugins)} plugins for APX model")

        return apx_info, plugins

    def get_capability_summary(self, apx_path: Path) -> Dict[str, Any]:
        """
        Get capability summary without full extraction.

        Args:
            apx_path: Path to APX file

        Returns:
            Summary dictionary with capabilities and metadata
        """
        if not apx_path.exists():
            raise FileNotFoundError(f"APX file not found: {apx_path}")

        # Read only the manifest from ZIP
        with zipfile.ZipFile(apx_path, 'r') as zf:
            if 'apx.yaml' not in zf.namelist():
                raise ValueError("Invalid APX: apx.yaml not found")

            with zf.open('apx.yaml') as f:
                manifest = yaml.safe_load(f)

        capabilities = manifest.get("capabilities", {}).get("provides", [])

        return {
            "name": manifest.get("name", "unknown"),
            "version": manifest.get("version", "unknown"),
            "type": manifest.get("type", "model"),
            "capabilities": capabilities,
            "manifest": manifest,
        }

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_dir}: {e}")

        self._temp_dirs.clear()

    def __del__(self):
        """Auto-cleanup on deletion."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False
