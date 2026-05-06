#!/usr/bin/env python
"""Run performance table evaluation"""
from app.evaluate import RecommenderEvaluator

print("Initializing evaluator...")
evaluator = RecommenderEvaluator()

print("Displaying performance table...\n")
metrics = evaluator.display_performance_table(k=5)

print("\nResults captured successfully!")
