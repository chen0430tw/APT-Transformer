#!/usr/bin/env python3
"""
Script to create a PR via the local proxy API
"""
import requests
import json

# Read the PR description
with open('PR_APT_2.0_DESCRIPTION.md', 'r', encoding='utf-8') as f:
    pr_body = f.read()

# PR data
pr_data = {
    "title": "APT 2.0: Complete Platform Architecture Refactoring",
    "head": "claude/review-project-structure-5A1Hl",
    "base": "main",
    "body": pr_body
}

# Try different API endpoints
api_endpoints = [
    "http://127.0.0.1:36226/api/repos/chen0430tw/APT-Transformer/pulls",
    "http://127.0.0.1:36226/repos/chen0430tw/APT-Transformer/pulls",
    "http://127.0.0.1:36226/api/v1/repos/chen0430tw/APT-Transformer/pulls",
]

for endpoint in api_endpoints:
    print(f"Trying endpoint: {endpoint}")
    try:
        response = requests.post(
            endpoint,
            json=pr_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text[:500]}")

        if response.status_code in [200, 201]:
            result = response.json()
            print(f"\n✅ PR created successfully!")
            print(f"PR URL: {result.get('html_url', 'N/A')}")
            print(f"PR Number: {result.get('number', 'N/A')}")
            exit(0)
    except Exception as e:
        print(f"Error: {e}")
    print()

print("❌ Failed to create PR via API. Please create manually via GitHub UI.")
print("PR description file: PR_APT_2.0_DESCRIPTION.md")
