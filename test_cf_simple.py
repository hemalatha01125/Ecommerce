#!/usr/bin/env python
"""Test CF with simplified implementation"""
import sys
sys.path.insert(0, '.')

from app.recommender import get_cf_scores, user_item

print("Testing simplified CF function...\n")

# Test with first user
user_id = str(user_item.index[0])
print(f"User ID: {user_id}")
print(f"Testing get_cf_scores()...")

cf_scores = get_cf_scores(user_id)

print(f"\nResults:")
print(f"  Total scores: {len(cf_scores)}")
print(f"  Empty? {cf_scores.empty}")

if not cf_scores.empty:
    print(f"  Min score: {cf_scores.min():.4f}")
    print(f"  Max score: {cf_scores.max():.4f}")
    print(f"  Mean score: {cf_scores.mean():.4f}")
    print(f"  Num zeros: {(cf_scores == 0).sum()}")
    
    print(f"\n  Top 5 scores:")
    for pid, score in cf_scores.nlargest(5).items():
        print(f"    {pid}: {score:.4f}")
    
    if (cf_scores == 0).sum() == 0:
        print("\n✓ SUCCESS - All scores are non-zero!")
    else:
        print(f"\n✗ ISSUE - {(cf_scores == 0).sum()} zero scores found")
else:
    print("\n✗ FAILED - CF returned empty scores")
