#!/usr/bin/env python
"""Test importing recommender module"""
import sys
import time

print(f"[{time.time()}] Starting...", flush=True)

try:
    print(f"[{time.time()}] Importing recommender...", flush=True)
    from app.recommender import get_cf_scores, get_cb_scores, hybrid_recommend, user_item
    print(f"[{time.time()}] ✓ recommender imported", flush=True)
    
    user_id = str(user_item.index[0])
    print(f"[{time.time()}] Testing get_cf_scores for {user_id}...", flush=True)
    
    cf_scores = get_cf_scores(user_id)
    print(f"[{time.time()}] ✓ CF scores returned: {len(cf_scores)} items", flush=True)
    
    if len(cf_scores) > 0:
        print(f"[{time.time()}] Top 3: {cf_scores.nlargest(3).to_dict()}", flush=True)
        print(f"[{time.time()}] ✓ SUCCESS - CF is working!", flush=True)
    else:
        print(f"[{time.time()}] ✗ EMPTY CF SCORES", flush=True)
    
except Exception as e:
    print(f"[{time.time()}] ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
