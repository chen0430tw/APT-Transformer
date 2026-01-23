#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Version Compatibility Checker

Implements semantic versioning compatibility checking for plugin engine requirements.
"""

from typing import Tuple
import re


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """
    Parse semantic version string.

    Args:
        version_str: Version string (e.g., "1.2.3")

    Returns:
        Tuple of (major, minor, patch)

    Raises:
        ValueError: If version string is invalid
    """
    # Remove "v" prefix if present
    version_str = version_str.lstrip('v')

    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")

    return tuple(map(int, match.groups()))


def version_compatible(current: str, requirement: str) -> bool:
    """
    Check if current version satisfies requirement.

    Supported operators:
    - ">=1.0.0"  - Greater than or equal
    - "~=1.2.0"  - Compatible version (1.2.x)
    - "==1.0.0"  - Exact match
    - "1.0.0"    - Defaults to >=

    Args:
        current: Current engine version
        requirement: Version requirement string

    Returns:
        True if requirement is satisfied

    Examples:
        >>> version_compatible("1.2.3", ">=1.0.0")
        True
        >>> version_compatible("1.2.3", "~=1.2.0")
        True
        >>> version_compatible("2.0.0", "~=1.2.0")
        False
    """
    requirement = requirement.strip()

    # Parse operator
    if requirement.startswith(">="):
        req_version = parse_version(requirement[2:])
        cur_version = parse_version(current)
        return cur_version >= req_version

    elif requirement.startswith("~="):
        # Compatible release: same major.minor, patch >= required
        req_version = parse_version(requirement[2:])
        cur_version = parse_version(current)
        return (
            cur_version[0] == req_version[0] and  # Same major
            cur_version[1] == req_version[1] and  # Same minor
            cur_version[2] >= req_version[2]      # Patch >=
        )

    elif requirement.startswith("=="):
        req_version = parse_version(requirement[2:])
        cur_version = parse_version(current)
        return cur_version == req_version

    else:
        # Default to >=
        req_version = parse_version(requirement)
        cur_version = parse_version(current)
        return cur_version >= req_version


class VersionChecker:
    """Engine version compatibility checker"""

    def __init__(self, engine_version: str = "1.0.0"):
        """
        Initialize version checker.

        Args:
            engine_version: Current engine version
        """
        self.engine_version = engine_version
        self._cache = {}  # Cache compatibility results

    def check_compatibility(self, requirement: str) -> Tuple[bool, str]:
        """
        Check if engine version meets requirement.

        Args:
            requirement: Version requirement string

        Returns:
            (is_compatible, reason)
        """
        # Check cache
        if requirement in self._cache:
            return self._cache[requirement]

        try:
            is_compat = version_compatible(self.engine_version, requirement)

            if is_compat:
                result = (True, "Compatible")
            else:
                result = (
                    False,
                    f"Engine {self.engine_version} does not meet {requirement}"
                )

            self._cache[requirement] = result
            return result

        except ValueError as e:
            result = (False, f"Invalid version format: {e}")
            self._cache[requirement] = result
            return result

    def check_plugin_compatibility(self, manifest: 'PluginManifest') -> Tuple[bool, str]:
        """
        Check if plugin is compatible with current engine version.

        Args:
            manifest: Plugin manifest

        Returns:
            (is_compatible, reason)
        """
        return self.check_compatibility(manifest.engine)
