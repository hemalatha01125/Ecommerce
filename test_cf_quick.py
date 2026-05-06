#!/usr/bin/env python
"""Quick test to verify CF scores are working"""
import sys
sys.path.insert(0, '.')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("Loading modules...")
from app.recommender import get_cf_scores, user_item, user_sim_df

print(f"User-item matrix shape: {user_item.shape}")
print(f"Number of users: {user_item.shape[0]}")

# Test with first user
test_user = str(user_item.index[0])
print(f"\nTesting CF scores for user: {test_user}")

cf_scores = get_cf_scores(test_user)
print(f"CF scores returned: {len(cf_scores)} items")

if len(cf_scores) > 0:
    print(f"Top 5 recommended items:")
    for pid, score in cf_scores.nlargest(5).items():
        print(f"  {pid}: {score:.4f}")
    print("✓ CF scores working!")
else:
    print("✗ CF scores are empty!")

print("\nDone!")
