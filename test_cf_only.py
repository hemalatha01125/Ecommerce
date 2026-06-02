#!/usr/bin/env python
"""Quick test of CF function output"""
from app.recommender import get_cf_scores, get_cb_scores, hybrid_recommend, user_item, df

print("=" * 60)
print("Testing CF, CB, and Hybrid")
print("=" * 60)

# Test with first user
user = list(user_item.index)[0]
product = df['product_id'].iloc[0]

print(f"\nTest User: {user}")
print(f"Test Product: {product}")
print()

cf_scores = get_cf_scores(user)
print(f"CF Scores: {len(cf_scores)} recommendations")
if not cf_scores.empty:
    print(cf_scores.head(5))
    print(f"Min: {cf_scores.min():.4f}, Max: {cf_scores.max():.4f}")
else:
    print("EMPTY - ALL ZEROS")

print()

cb_scores = get_cb_scores(product)
print(f"CB Scores: {len(cb_scores)} recommendations")
if not cb_scores.empty:
    print(cb_scores.head(5))
    print(f"Min: {cb_scores.min():.4f}, Max: {cb_scores.max():.4f}")
else:
    print("EMPTY")

print()

hybrid = hybrid_recommend(user, product, top_n=5)
print(f"Hybrid Recommendations: {len(hybrid)} items")
if hybrid:
    for i, rec in enumerate(hybrid):
        print(f"  {i+1}. {rec['product_name'][:50]} - Rating: {rec['rating']}")
else:
    print("NONE")

print("\n" + "=" * 60)
