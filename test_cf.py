from app.recommender import get_cf_scores, user_item, get_cf_scores, user_sim_df
from app.evaluate import RecommenderEvaluator
import pandas as pd

print("=" * 80)
print("DEBUG: CF SCORES ANALYSIS")
print("=" * 80)

# Check what's in user_item
print(f"\nUser-item matrix shape: {user_item.shape}")
print(f"Number of unique users: {user_item.shape[0]}")
print(f"Sample user IDs in matrix: {list(user_item.index)[:5]}")
print(f"Type of user IDs: {type(user_item.index[0])}")

# Test get_cf_scores with a real user
test_user = str(user_item.index[0])
print(f"\n--- Testing with user {test_user} ---")
scores = get_cf_scores(test_user)
print(f"Number of CF scores returned: {len(scores)}")
if len(scores) > 0:
    print(f"Max score: {scores.max()}")
    print(f"Top 5 scores: {scores.nlargest(5).to_dict()}")
else:
    print("CF Scores are EMPTY!")

# Check similar users
print(f"\nSimilar users to {test_user}:")
similar_users = user_sim_df[test_user].sort_values(ascending=False)[1:6]
print(similar_users)

# Check already rated items for this user
print(f"\nAlready rated items for user {test_user}:")
rated_items = user_item.loc[test_user].dropna()
print(f"Number of rated items: {len(rated_items)}")
if len(rated_items) > 0:
    print(f"Sample rated items: {list(rated_items.index)[:5]}")

# Now test with evaluate
print("\n" + "=" * 80)
print("DEBUG: EVALUATE ALGORITHM")
print("=" * 80)

evaluator = RecommenderEvaluator(test_size=0.2, random_state=42)

# Get a test user
test_users = list(evaluator.test_data['user_id'].unique())[:1]
print(f"\nTest users sample: {test_users}")

# Check if test user is in the user_item matrix
for test_user in test_users:
    print(f"\nUser {test_user} in user_item.index? {test_user in user_item.index}")
    
    # Try to get CF scores
    cf_scores = get_cf_scores(test_user)
    print(f"CF scores returned: {len(cf_scores)}")
    if len(cf_scores) > 0:
        print(f"Top scores: {cf_scores.nlargest(5).to_dict()}")
