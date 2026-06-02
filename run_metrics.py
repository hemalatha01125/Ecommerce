#!/usr/bin/env python
"""Run performance table evaluation"""
from app.evaluate import RecommenderEvaluator

print("Initializing evaluator...")
evaluator = RecommenderEvaluator()

print("Running full evaluation...\n")
all_results = evaluator.run_full_evaluation(k_values=[5, 10])

print("\nResults captured successfully!")
