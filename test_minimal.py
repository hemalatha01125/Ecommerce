#!/usr/bin/env python
"""Minimal test - just check imports"""
import sys
import time

print(f"[{time.time()}] Python started", flush=True)

try:
    print(f"[{time.time()}] Importing pandas...", flush=True)
    import pandas as pd
    print(f"[{time.time()}] ✓ pandas imported", flush=True)
    
    print(f"[{time.time()}] Importing sklearn...", flush=True)
    from sklearn.metrics.pairwise import cosine_similarity
    print(f"[{time.time()}] ✓ sklearn imported", flush=True)
    
    print(f"[{time.time()}] Loading CSV...", flush=True)
    df = pd.read_csv("data/raw/amazon.csv")
    print(f"[{time.time()}] ✓ CSV loaded: {df.shape}", flush=True)
    
    print(f"[{time.time()}] Building user-item matrix...", flush=True)
    df_cleaned = df[['user_id', 'product_id', 'rating']].dropna()
    df_cleaned['user_id'] = df_cleaned['user_id'].astype(str)
    df_cleaned['product_id'] = df_cleaned['product_id'].astype(str)
    df_cleaned['rating'] = pd.to_numeric(df_cleaned['rating'], errors='coerce').dropna()
    
    user_item = df_cleaned.pivot_table(index='user_id', columns='product_id', values='rating')
    print(f"[{time.time()}] ✓ Matrix built: {user_item.shape}", flush=True)
    
    print(f"[{time.time()}] Computing similarity...", flush=True)
    user_item_filled = user_item.fillna(0)
    user_sim = cosine_similarity(user_item_filled)
    print(f"[{time.time()}] ✓ Similarity computed: {user_sim.shape}", flush=True)
    
    print(f"[{time.time()}] SUCCESS!", flush=True)
    
except Exception as e:
    print(f"[{time.time()}] ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
