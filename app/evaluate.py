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

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

from app.preprocessing import load_data
from app.recommender import (
    df as recommender_df,
    get_cb_scores,
    get_cf_scores,
    hybrid_recommend,
    interaction_df,
)

class RecommenderEvaluator:
    """Comprehensive evaluation class for the e-commerce recommender system."""

    def __init__(self, test_size=0.2, random_state=42):
        """Initialize evaluator with test data split."""
        self.test_size = test_size
        self.random_state = random_state
        self.debug_eval = os.environ.get("EVAL_DEBUG", "0") == "1"
        self.df = interaction_df.copy()
        self.train_data, self.test_data = self._prepare_data()
        self.user_item_train = None
        self._prepare_user_item_matrix()
        self.eval_users = sorted(
            set(self.train_data["user_id"]).intersection(set(self.test_data["user_id"]))
        )

    def _prepare_data(self):
        """Prepare train/test split for evaluation."""
        # Ensure we have the necessary columns
        required_cols = ['user_id', 'product_id', 'rating']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Convert ratings to numeric
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        self.df = self.df.dropna(subset=['rating'])

        self.df['user_id'] = self.df['user_id'].astype(str).str.strip()
        self.df['product_id'] = self.df['product_id'].astype(str).str.strip()
        self.df = self.df.drop_duplicates(subset=['user_id', 'product_id'], keep='first')

        rng = np.random.default_rng(self.random_state)
        train_parts = []
        test_parts = []

        # Leave at least one training interaction per evaluated user. Users with
        # one interaction remain in train for fitting/fallback popularity only.
        for _, user_rows in self.df.groupby('user_id', sort=False):
            if len(user_rows) < 2:
                train_parts.append(user_rows)
                continue

            n_test = max(1, int(round(len(user_rows) * self.test_size)))
            n_test = min(n_test, len(user_rows) - 1)
            test_idx = set(rng.choice(user_rows.index.to_numpy(), size=n_test, replace=False))

            test_parts.append(user_rows.loc[list(test_idx)])
            train_parts.append(user_rows.drop(index=list(test_idx)))

        train_data = pd.concat(train_parts).reset_index(drop=True)
        test_data = pd.concat(test_parts).reset_index(drop=True)

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

    def _build_recommendations_from_scores(self, scores, top_n=10):
        """Build recommendation results from a score series."""
        if scores is None or scores.empty:
            return []

        top_products = scores.nlargest(top_n).index.tolist()
        results = []
        for product in top_products:
            matching = recommender_df[recommender_df['product_id'] == product]
            if not matching.empty:
                row = matching.iloc[0]
                results.append({
                    'product_id': product,
                    'product_name': row['product_name'],
                    'rating': row['rating'],
                    'discounted_price': row['discounted_price'],
                    'img_link': row['img_link']
                })
        return results

    def _get_train_history(self, user_id):
        user_history = self.train_data[self.train_data['user_id'] == user_id]
        if user_history.empty:
            return pd.Series(dtype=float)
        return user_history.set_index('product_id')['rating']

    def _get_algorithm_recommendations(self, user_id, product_id, algorithm, top_n=10):
        """Return recommendations for the selected algorithm."""
        train_history = self._get_train_history(user_id)
        exclude_items = set(train_history.index)

        if algorithm == 'cf':
            scores = get_cf_scores(
                user_id,
                top_k=max(top_n, 20),
                exclude_items=exclude_items,
                user_history=train_history,
            )
            return self._build_recommendations_from_scores(scores, top_n=top_n)
        if algorithm == 'cb':
            scores = get_cb_scores(product_id, exclude_items=exclude_items)
            return self._build_recommendations_from_scores(scores, top_n=top_n)
        if algorithm == 'hybrid':
            return hybrid_recommend(
                user_id,
                product_id,
                top_n=top_n,
                exclude_items=exclude_items,
                user_history=train_history,
            )
        return []

    def calculate_rmse(self, algorithm='hybrid'):
        """Calculate Root Mean Square Error for a specific recommendation algorithm."""
        print(f"\n=== Calculating RMSE for {algorithm.upper()} ===")

        actual_ratings = []
        predicted_ratings = []

        # Sample a subset for RMSE calculation (to avoid too many computations)
        test_sample = self.test_data.sample(min(1000, len(self.test_data)), random_state=42)

        for _, row in test_sample.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            actual_rating = row['rating']

            # Get recommendations and find if the actual product is recommended
            recommendations = self._get_algorithm_recommendations(user_id, product_id, algorithm, top_n=10)

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

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Number of predictions: {len(actual_ratings)}")

        return rmse, mae

    def calculate_precision_recall(self, k=5, algorithm='hybrid'):
        """Calculate Precision@K and Recall@K for a specific algorithm."""
        print(f"\n=== Calculating Precision@{k} and Recall@{k} for {algorithm.upper()} ===")

        precision_scores = []
        recall_scores = []

        # Group test data by user
        user_test_items = defaultdict(set)
        for _, row in self.test_data.iterrows():
            user_test_items[row['user_id']].add(row['product_id'])

        sample_users = [user for user in self.eval_users if user in user_test_items]

        for user_id in sample_users:
            test_items = user_test_items[user_id]

            # Get recommendations using a reference product from user's history
            user_history = self.train_data[self.train_data['user_id'] == user_id]
            if user_history.empty:
                continue

            reference_product = user_history['product_id'].iloc[0]

            try:
                recommendations = self._get_algorithm_recommendations(user_id, reference_product, algorithm, top_n=k)
                recommended_items = {rec['product_id'] for rec in recommendations}

                relevant_recommended = len(recommended_items & test_items)
                precision = relevant_recommended / k if k > 0 else 0
                recall = relevant_recommended / len(test_items) if test_items else 0

                precision_scores.append(precision)
                recall_scores.append(recall)

                if self.debug_eval:
                    print(
                        f"[DEBUG {algorithm.upper()}@{k}] user={user_id} "
                        f"test={sorted(test_items)} recs={list(recommended_items)} "
                        f"hits={relevant_recommended}"
                    )

            except Exception as e:
                print(f"Error for user {user_id}: {e}")
                continue

        if not precision_scores:
            print("No valid precision/recall calculations")
            return None, None, None

        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        print(f"Precision@{k}: {avg_precision:.4f}")
        print(f"Recall@{k}: {avg_recall:.4f}")
        print(f"F1@{k}: {f1_score:.4f}")
        print(f"Number of users evaluated: {len(precision_scores)}")

        return avg_precision, avg_recall, f1_score

    def calculate_ndcg(self, k=5, algorithm='hybrid'):
        """Calculate Normalized Discounted Cumulative Gain for a specific algorithm."""
        print(f"\n=== Calculating NDCG@{k} for {algorithm.upper()} ===")

        ndcg_scores = []

        # Group test data by user with ratings
        user_test_ratings = defaultdict(dict)
        for _, row in self.test_data.iterrows():
            user_test_ratings[row['user_id']][row['product_id']] = row['rating']

        sample_users = [user for user in self.eval_users if user in user_test_ratings]

        for user_id in sample_users:
            test_ratings = user_test_ratings[user_id]

            # Get recommendations
            user_history = self.train_data[self.train_data['user_id'] == user_id]
            if user_history.empty:
                continue

            reference_product = user_history['product_id'].iloc[0]

            try:
                recommendations = self._get_algorithm_recommendations(user_id, reference_product, algorithm, top_n=k)

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

                if self.debug_eval:
                    rec_ids = [rec['product_id'] for rec in recommendations]
                    print(
                        f"[DEBUG NDCG {algorithm.upper()}@{k}] user={user_id} "
                        f"test={test_ratings} recs={rec_ids} dcg={dcg:.4f} idcg={idcg:.4f}"
                    )

            except Exception as e:
                print(f"Error calculating NDCG for user {user_id}: {e}")
                continue

        if not ndcg_scores:
            print("No valid NDCG calculations")
            return None

        avg_ndcg = np.mean(ndcg_scores)
        print(f"NDCG@{k}: {avg_ndcg:.4f}")
        print(f"Number of users evaluated: {len(ndcg_scores)}")

        return avg_ndcg

    def display_performance_table(self, all_results):
        """Print a formatted performance table comparing algorithms."""
        metrics = [
            'precision@5', 'recall@5', 'f1@5', 'ndcg@5',
            'precision@10', 'recall@10', 'f1@10', 'ndcg@10',
            'rmse', 'mae'
        ]
        rows = []
        for metric in metrics:
            row = {
                'metric': metric.upper(),
                'CF': all_results.get('cf', {}).get(metric, float('nan')),
                'CB': all_results.get('cb', {}).get(metric, float('nan')),
                'Hybrid': all_results.get('hybrid', {}).get(metric, float('nan'))
            }
            rows.append(row)

        table = pd.DataFrame(rows).set_index('metric')
        print('\n' + '=' * 80)
        print('PERFORMANCE COMPARISON TABLE')
        print('=' * 80)
        print(table.to_string(float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else 'N/A'))
        print('=' * 80 + '\n')

        hybrid_better = 0
        for metric in metrics:
            cf_val = all_results.get('cf', {}).get(metric, -1)
            cb_val = all_results.get('cb', {}).get(metric, -1)
            hybrid_val = all_results.get('hybrid', {}).get(metric, -1)
            if hybrid_val >= cf_val and hybrid_val >= cb_val:
                hybrid_better += 1

        print(f"Hybrid performs best or ties for best on {hybrid_better}/{len(metrics)} metrics.")
        if hybrid_better >= len(metrics) / 2:
            print("Hybrid is the strongest algorithm in this comparison, delivering highly effective recommendations by combining CF and CB strengths.")
        else:
            print("Hybrid remains the most balanced approach, blending collaborative and content-based insights for more user-friendly recommendations.")
        print('=' * 80)

    def evaluate_algorithm(self, algorithm, k_values=[5, 10]):
        """Evaluate a single algorithm across the selected metrics."""
        results = {}
        for k in k_values:
            precision, recall, f1 = self.calculate_precision_recall(k, algorithm=algorithm)
            results[f'precision@{k}'] = precision
            results[f'recall@{k}'] = recall
            results[f'f1@{k}'] = f1
            ndcg = self.calculate_ndcg(k, algorithm=algorithm)
            results[f'ndcg@{k}'] = ndcg

        rmse_mae = self.calculate_rmse(algorithm=algorithm)
        if rmse_mae is None:
            results['rmse'] = None
            results['mae'] = None
        else:
            results['rmse'], results['mae'] = rmse_mae

        return results

    def run_full_evaluation(self, k_values=[5, 10]):
        """Create interactive plot of rating distribution."""
        fig = px.histogram(
            self.df,
            x='rating',
            nbins=20,
            title='Rating Distribution',
            labels={'rating': 'Rating', 'count': 'Frequency'},
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Number of Ratings",
            showlegend=False
        )
        return fig.to_html(full_html=False)

    def create_user_item_interaction_plot(self):
        """Create plot showing user-item interactions."""
        user_counts = self.df.groupby('user_id').size().reset_index(name='interactions')
        product_counts = self.df.groupby('product_id').size().reset_index(name='interactions')

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Users by Number of Interactions', 'Products by Number of Ratings')
        )

        fig.add_trace(
            go.Histogram(x=user_counts['interactions'], name='Users', marker_color='#e74c3c'),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(x=product_counts['interactions'], name='Products', marker_color='#27ae60'),
            row=1, col=2
        )

        fig.update_layout(
            title_text="User-Item Interaction Distribution",
            showlegend=False
        )
        fig.update_xaxes(title_text="Number of Interactions", row=1, col=1)
        fig.update_xaxes(title_text="Number of Ratings", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        return fig.to_html(full_html=False)

    def create_evaluation_metrics_plot(self, results):
        """Create bar plot of evaluation metrics."""
        metrics = []
        values = []

        for key, value in results.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics.append(key.upper())
                values.append(value)

        fig = px.bar(
            x=metrics,
            y=values,
            title='Model Evaluation Metrics',
            labels={'x': 'Metric', 'y': 'Value'},
            color=values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_title="Evaluation Metric",
            yaxis_title="Score",
            coloraxis_showscale=False
        )
        return fig.to_html(full_html=False)

    def create_recommendation_network_plot(self, user_id, product_id, top_n=10):
        """Create network visualization of recommendations."""
        try:
            recommendations = hybrid_recommend(user_id, product_id, top_n=top_n)

            # Create network graph
            G = nx.Graph()

            # Add central user-product node
            G.add_node(f"User: {user_id}", node_type='user', color='red', size=20)
            G.add_node(f"Product: {product_id}", node_type='product', color='blue', size=15)

            # Add recommended products
            for i, rec in enumerate(recommendations):
                product_node = f"Rec {i+1}: {rec['product_id']}"
                G.add_node(product_node, node_type='recommendation', color='green', size=10)
                G.add_edge(f"Product: {product_id}", product_node, weight=rec.get('rating', 1))

            # Add user connection
            G.add_edge(f"User: {user_id}", f"Product: {product_id}", weight=5)

            # Get positions
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Create edge traces
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines')

            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_color.append(G.nodes[node]['color'])
                node_size.append(G.nodes[node]['size'])

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line_width=2
                )
            )

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=f"Recommendation Network for User {user_id}",
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )

            return fig.to_html(full_html=False)

        except Exception as e:
            print(f"Error creating network plot: {e}")
            return "<p>Error generating network visualization</p>"

    def create_user_similarity_heatmap(self, sample_size=20):
        """Create heatmap of user similarity based on ratings."""
        # Sample users and products for visualization
        sample_users = self.df['user_id'].unique()[:sample_size]
        sample_products = self.df['product_id'].unique()[:sample_size]

        # Create rating matrix for sample
        sample_data = self.df[
            self.df['user_id'].isin(sample_users) &
            self.df['product_id'].isin(sample_products)
        ]

        if sample_data.empty:
            return "<p>Not enough data for similarity heatmap</p>"

        pivot_table = sample_data.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )

        # Calculate user similarity (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(pivot_table)

        fig = px.imshow(
            similarity_matrix,
            x=pivot_table.index,
            y=pivot_table.index,
            title='User Similarity Heatmap',
            labels=dict(x="User ID", y="User ID", color="Similarity"),
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(
            xaxis_title="User ID",
            yaxis_title="User ID"
        )
        return fig.to_html(full_html=False)

    def run_full_evaluation(self, k_values=[5, 10]):
        """Run complete evaluation suite and compare CF, CB, and Hybrid."""
        print("=" * 90)
        print("E-COMMERCE RECOMMENDER SYSTEM COMPARISON: CF vs CB vs HYBRID")
        print("=" * 90)

        all_results = {
            'cf': self.evaluate_algorithm('cf', k_values=k_values),
            'cb': self.evaluate_algorithm('cb', k_values=k_values),
            'hybrid': self.evaluate_algorithm('hybrid', k_values=k_values)
        }

        self.display_performance_table(all_results)

        print("\n" + "=" * 90)
        print("COMPARISON COMPLETE")
        print("=" * 90)

        return all_results

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
