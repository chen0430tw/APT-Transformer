#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test profile loader without importing the full apt package
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import only the profile_loader module
from apt.core.config.profile_loader import load_profile, list_profiles

print("✓ Profile loader imported successfully\n")

# List profiles
print("Available profiles:")
profiles = list_profiles()
for p in profiles:
    print(f"  - {p}")
print()

# Load and test each profile
for profile_name in profiles:
    print(f"Testing {profile_name}:")
    try:
        config = load_profile(profile_name)
        print(f"  ✓ Profile: {config.profile.name}")
        print(f"  ✓ Description: {config.profile.description[:50]}...")
        print(f"  ✓ Hidden size: {config.model.hidden_size}")
        print(f"  ✓ Batch size: {config.training.batch_size}")
        print(f"  ✓ VGPU enabled: {config.vgpu.enabled}")
        print()
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("✓ All profile tests completed")
