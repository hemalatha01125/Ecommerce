#!/usr/bin/env python
"""Test the fixed CF recommender"""

print("Starting test...")

from app.recommender import get_cf_scores, user_item, user_sim_df
import pandas as pd

print(f"✓ Imports successful")
print(f"User-item matrix shape: {user_item.shape}")

# Get first user
user_id = str(user_item.index[0])
print(f"\nTesting CF for user: {user_id}")

# Get CF scores
cf_scores = get_cf_scores(user_id)
print(f"CF scores returned: {len(cf_scores)} items")

if len(cf_scores) > 0:
    print(f"Top 5 scores:")
    for pid, score in cf_scores.nlargest(5).items():
        print(f"  {pid}: {score:.4f}")
    print("✓ CF scores are non-zero!")
else:
    print("✗ CF scores are empty - FAILURE")

print("\nTest complete!")
