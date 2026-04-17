import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from app.preprocessing import load_data
from app.recommender import hybrid_recommend

class RecommenderEvaluator:
    """Comprehensive evaluation class for the e-commerce recommender system."""

    def __init__(self, test_size=0.2, random_state=42):
        """Initialize evaluator with test data split."""
        self.test_size = test_size
        self.random_state = random_state
        self.df = load_data()
        self.train_data, self.test_data = self._prepare_data()
        self.user_item_train = None
        self._prepare_user_item_matrix()

    def _prepare_data(self):
        """Prepare train/test split for evaluation."""
        # Ensure we have the necessary columns
        required_cols = ['user_id', 'product_id', 'rating']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Convert ratings to numeric
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        self.df = self.df.dropna(subset=['rating'])

        # Convert user_id and product_id to string for consistency
        self.df['user_id'] = self.df['user_id'].astype(str)
        self.df['product_id'] = self.df['product_id'].astype(str)

        # Split data (remove stratification if it causes issues)
        try:
            train_data, test_data = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.df['user_id'] if len(self.df['user_id'].unique()) > 1 else None
            )
        except ValueError:
            # Fallback to simple split if stratification fails
            print("Warning: Using simple train/test split (stratification failed)")
            train_data, test_data = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state
            )

        print(f"Train set: {len(train_data)} interactions")
        print(f"Test set: {len(test_data)} interactions")
        print(f"Unique users in train: {train_data['user_id'].nunique()}")
        print(f"Unique users in test: {test_data['user_id'].nunique()}")

        return train_data, test_data

    def _prepare_user_item_matrix(self):
        """Create user-item matrix from training data."""
        self.user_item_train = self.train_data.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            aggfunc='mean'
        ).fillna(0)

    def calculate_rmse(self):
        """Calculate Root Mean Square Error."""
        print("\n=== Calculating RMSE ===")

        actual_ratings = []
        predicted_ratings = []

        # Sample a subset for RMSE calculation (to avoid too many computations)
        test_sample = self.test_data.sample(min(1000, len(self.test_data)), random_state=42)

        for _, row in test_sample.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            actual_rating = row['rating']

            # Get recommendations and find if the actual product is recommended
            recommendations = hybrid_recommend(user_id, product_id, top_n=10)

            # For RMSE, we need predicted rating for the actual product
            # Since our system returns recommendations, we'll use the average rating of recommended items
            if recommendations:
                pred_rating = np.mean([rec.get('rating', 0) for rec in recommendations if rec.get('rating')])
            else:
                pred_rating = self.train_data['rating'].mean()  # Global mean fallback

            actual_ratings.append(actual_rating)
            predicted_ratings.append(pred_rating)

        if not actual_ratings:
            print("No valid predictions for RMSE calculation")
            return None

        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        mae = mean_absolute_error(actual_ratings, predicted_ratings)

        print(".4f")
        print(".4f")
        print(f"Number of predictions: {len(actual_ratings)}")

        return rmse, mae

    def calculate_precision_recall(self, k=5):
        """Calculate Precision@K and Recall@K."""
        print(f"\n=== Calculating Precision@{k} and Recall@{k} ===")

        precision_scores = []
        recall_scores = []

        # Group test data by user
        user_test_items = defaultdict(set)
        for _, row in self.test_data.iterrows():
            user_test_items[row['user_id']].add(row['product_id'])

        # Sample users for evaluation
        test_users = list(user_test_items.keys())
        sample_users = np.random.choice(test_users, min(100, len(test_users)), replace=False)

        for user_id in sample_users:
            test_items = user_test_items[user_id]

            # Get recommendations using a random product from user's history
            user_history = self.train_data[self.train_data['user_id'] == user_id]
            if user_history.empty:
                continue

            # Use the most recent product or a random one as reference
            reference_product = user_history['product_id'].iloc[0]

            try:
                recommendations = hybrid_recommend(user_id, reference_product, top_n=k)
                recommended_items = {rec['product_id'] for rec in recommendations}

                # Calculate precision and recall
                relevant_recommended = len(recommended_items & test_items)
                precision = relevant_recommended / k if k > 0 else 0
                recall = relevant_recommended / len(test_items) if test_items else 0

                precision_scores.append(precision)
                recall_scores.append(recall)

            except Exception as e:
                print(f"Error for user {user_id}: {e}")
                continue

        if not precision_scores:
            print("No valid precision/recall calculations")
            return None, None, None

        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        print(".4f")
        print(".4f")
        print(".4f")
        print(f"Number of users evaluated: {len(precision_scores)}")

        return avg_precision, avg_recall, f1_score

    def calculate_ndcg(self, k=5):
        """Calculate Normalized Discounted Cumulative Gain."""
        print(f"\n=== Calculating NDCG@{k} ===")

        ndcg_scores = []

        # Group test data by user with ratings
        user_test_ratings = defaultdict(dict)
        for _, row in self.test_data.iterrows():
            user_test_ratings[row['user_id']][row['product_id']] = row['rating']

        test_users = list(user_test_ratings.keys())
        sample_users = np.random.choice(test_users, min(50, len(test_users)), replace=False)

        for user_id in sample_users:
            test_ratings = user_test_ratings[user_id]

            # Get recommendations
            user_history = self.train_data[self.train_data['user_id'] == user_id]
            if user_history.empty:
                continue

            reference_product = user_history['product_id'].iloc[0]

            try:
                recommendations = hybrid_recommend(user_id, reference_product, top_n=k)

                # Calculate DCG
                dcg = 0
                for i, rec in enumerate(recommendations):
                    relevance = test_ratings.get(rec['product_id'], 0)
                    dcg += relevance / np.log2(i + 2)  # i+2 because positions start from 1

                # Calculate IDCG (ideal DCG - best possible ranking)
                ideal_relevances = sorted(test_ratings.values(), reverse=True)[:k]
                idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)

            except Exception as e:
                print(f"Error calculating NDCG for user {user_id}: {e}")
                continue

        if not ndcg_scores:
            print("No valid NDCG calculations")
            return None

        avg_ndcg = np.mean(ndcg_scores)
        print(".4f")
        print(f"Number of users evaluated: {len(ndcg_scores)}")

        return avg_ndcg

    def run_full_evaluation(self, k_values=[5, 10]):
        """Run complete evaluation suite."""
        print("=" * 60)
        print("E-COMMERCE RECOMMENDER SYSTEM EVALUATION")
        print("=" * 60)

        results = {}

        # RMSE and MAE
        rmse_result = self.calculate_rmse()
        if rmse_result:
            results['rmse'], results['mae'] = rmse_result

        # Precision, Recall, F1 for different k values
        for k in k_values:
            precision, recall, f1 = self.calculate_precision_recall(k)
            if precision is not None:
                results[f'precision@{k}'] = precision
                results[f'recall@{k}'] = recall
                results[f'f1@{k}'] = f1

            ndcg = self.calculate_ndcg(k)
            if ndcg is not None:
                results[f'ndcg@{k}'] = ndcg

        # Summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        for metric, value in results.items():
            print("20")

        return results

def main():
    """Main evaluation function."""
    try:
        evaluator = RecommenderEvaluator(test_size=0.2, random_state=42)
        results = evaluator.run_full_evaluation(k_values=[5, 10])

        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)

        return results

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()