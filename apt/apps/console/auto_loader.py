#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automatic Plugin Loader

Automatically loads plugins based on model capabilities detected via APX.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

from apt.apps.console.plugin_standards import PluginBase
from apt.apps.console.capability_plugin_map import (
    get_recommended_plugins,
    check_plugin_requirements,
    get_plugin_score,
)

logger = logging.getLogger(__name__)


class AutoPluginLoader:
    """Automatic plugin loader based on model capabilities"""

    def __init__(self, plugin_registry: Dict[str, type]):
        """
        Initialize auto-loader.

        Args:
            plugin_registry: Plugin registry {name: PluginClass}
        """
        self.plugin_registry = plugin_registry

    def analyze_model(self, capabilities: List[str]) -> Dict[str, Any]:
        """
        Analyze model capabilities and generate plugin recommendations.

        Args:
            capabilities: List of model capabilities

        Returns:
            Analysis results dictionary with:
            - capabilities: Detected capabilities
            - recommended_plugins: All recommended plugins
            - available_plugins: Plugins that can be loaded
            - unavailable_plugins: Plugins that cannot be loaded
            - plugin_scores: Relevance scores for each plugin
        """
        # Get recommended plugins
        recommended = get_recommended_plugins(capabilities)

        # Check each plugin's requirements and availability
        available_plugins = []
        unavailable_plugins = []
        plugin_scores = {}

        for plugin_name in recommended:
            # Check requirements
            satisfied, reason = check_plugin_requirements(plugin_name, capabilities)

            # Calculate relevance score
            score = get_plugin_score(plugin_name, capabilities)
            plugin_scores[plugin_name] = score

            if satisfied and plugin_name in self.plugin_registry:
                available_plugins.append({
                    "name": plugin_name,
                    "reason": reason,
                    "score": score,
                })
            else:
                unavailable_plugins.append({
                    "name": plugin_name,
                    "reason": reason if not satisfied else "Plugin not registered",
                    "score": score,
                })

        # Sort available plugins by score (descending)
        available_plugins.sort(key=lambda x: x["score"], reverse=True)

        return {
            "capabilities": capabilities,
            "recommended_plugins": recommended,
            "available_plugins": available_plugins,
            "unavailable_plugins": unavailable_plugins,
            "plugin_scores": plugin_scores,
        }

    def load_for_capabilities(
        self,
        capabilities: List[str],
        auto_enable: bool = True,
        dry_run: bool = False,
        score_threshold: float = 0.0,
    ) -> List[PluginBase]:
        """
        Load plugins for model with specified capabilities.

        Args:
            capabilities: Model capabilities
            auto_enable: Whether to auto-enable recommended plugins
            dry_run: Only analyze, don't instantiate plugins
            score_threshold: Minimum relevance score to load (0.0-1.0)

        Returns:
            List of loaded plugin instances
        """
        analysis = self.analyze_model(capabilities)

        logger.info(f"Model capabilities: {capabilities}")
        logger.info(f"Recommended plugins: {analysis['recommended_plugins']}")

        if dry_run:
            return []

        loaded_plugins = []

        if auto_enable:
            for plugin_info in analysis['available_plugins']:
                plugin_name = plugin_info['name']
                score = plugin_info['score']

                # Check score threshold
                if score < score_threshold:
                    logger.debug(
                        f"Skipping {plugin_name} (score {score:.2f} < threshold {score_threshold:.2f})"
                    )
                    continue

                try:
                    # Instantiate plugin
                    plugin_class = self.plugin_registry[plugin_name]
                    plugin = plugin_class()
                    loaded_plugins.append(plugin)

                    logger.info(
                        f"Auto-loaded plugin: {plugin_name} (score: {score:.2f})"
                    )

                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {e}", exc_info=True)

        return loaded_plugins

    def get_recommendations_report(
        self,
        capabilities: List[str],
        format: str = "text"
    ) -> str:
        """
        Generate human-readable recommendations report.

        Args:
            capabilities: Model capabilities
            format: Output format ("text" or "json")

        Returns:
            Formatted report string
        """
        analysis = self.analyze_model(capabilities)

        if format == "json":
            import json
            return json.dumps(analysis, indent=2)

        # Text format
        lines = []
        lines.append("=" * 60)
        lines.append("Plugin Recommendation Report")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Model Capabilities: {', '.join(capabilities) if capabilities else 'none'}")
        lines.append("")

        if analysis['available_plugins']:
            lines.append("Available Plugins (Ready to Load):")
            for p in analysis['available_plugins']:
                lines.append(f"  ✅ {p['name']:30s} (score: {p['score']:.2f})")
                lines.append(f"      {p['reason']}")
        else:
            lines.append("Available Plugins: none")

        lines.append("")

        if analysis['unavailable_plugins']:
            lines.append("Unavailable Plugins:")
            for p in analysis['unavailable_plugins']:
                lines.append(f"  ❌ {p['name']:30s}")
                lines.append(f"      {p['reason']}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
