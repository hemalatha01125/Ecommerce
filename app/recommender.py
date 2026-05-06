import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/amazon.csv")

# Keep required columns
df = df[['user_id', 'product_id', 'product_name', 'category', 'about_product', 'rating', 'discounted_price', 'img_link']]

# Drop missing values
df = df.dropna()

# Convert rating
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])

# Convert user_id and product_id to string for consistency
df['user_id'] = df['user_id'].astype(str)
df['product_id'] = df['product_id'].astype(str)

# Reset index to ensure sequential indexing (0, 1, 2, ...) for proper matrix access
df = df.reset_index(drop=True)

# Create mapping from product_id to matrix row index for O(1) lookups
product_id_to_idx = pd.Series(df.index, index=df['product_id']).to_dict()

# =========================
# CONTENT-BASED FILTERING
# =========================

# Combine text
df['content'] = df['product_name'] + " " + df['category'] + " " + df['about_product']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['content'])

# Cosine similarity
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# =========================
# COLLABORATIVE FILTERING
# =========================

# User-item matrix
user_item = df.pivot_table(index='user_id', columns='product_id', values='rating')

# Fill missing
user_item_filled = user_item.fillna(0)

# User similarity
user_sim = cosine_similarity(user_item_filled)

# Convert to DataFrame
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

# =========================
# CF FUNCTION
# =========================
def get_cf_scores(user_id, n_similar_users=15):
    """
    Collaborative Filtering recommendation scores for a user.

    - Uses top_k similar users (15 by default)
    - Falls back for cold start users
    - Handles empty similarity results
    - Ensures non-zero prediction scores
    """

    if user_id not in user_item.index:
        return _get_popular_items()

    try:
        similarities = user_sim_df.loc[user_id].sort_values(ascending=False)
        similar_user_ids = similarities.index[1:n_similar_users + 1].tolist()
    except Exception:
        return _get_popular_items()

    if not similar_user_ids:
        return _get_popular_items()

    scores = pd.Series(dtype=float)

    for similar_user in similar_user_ids:
        try:
            sim_weight = user_sim_df.loc[user_id, similar_user]
            if sim_weight <= 0:
                continue

            user_ratings = user_item.loc[similar_user]
            rated_values = user_ratings[user_ratings > 0]
            if rated_values.empty:
                continue

            scores = scores.add(rated_values * sim_weight, fill_value=0)
        except Exception:
            continue

    if scores.empty:
        return _get_popular_items()

    scores = scores[scores > 0]
    if scores.empty:
        return _get_popular_items()

    return scores


def _get_popular_items():
    """Global fallback recommendations using average product ratings."""
    try:
        avg_ratings = user_item.mean(axis=0)
        avg_ratings = avg_ratings[avg_ratings > 0]
        if not avg_ratings.empty:
            return avg_ratings
        return user_item.sum(axis=0)[user_item.sum(axis=0) > 0]
    except Exception:
        return pd.Series(dtype=float)


# =========================
# CB FUNCTION - SIMPLIFIED
# =========================
def get_cb_scores(product_id):
    """
    Content-based similarity scores for a product.
    
    Args:
        product_id: Target product ID
    
    Returns:
        pd.Series: Similarity scores for similar products
    """
    if product_id not in product_id_to_idx:
        return pd.Series(dtype=float)
    
    try:
        idx = product_id_to_idx[product_id]
        
        # Get similarity scores from pre-computed matrix
        sim_scores = pd.Series(content_sim[idx], index=df['product_id'])
        
        # Remove the product itself
        sim_scores = sim_scores.drop(product_id, errors='ignore')
        
        # Keep only positive similarities
        sim_scores = sim_scores[sim_scores > 0]
        
        return sim_scores if not sim_scores.empty else pd.Series(dtype=float)
    
    except:
        return pd.Series(dtype=float)


# =========================
# HYBRID RECOMMENDER - SIMPLIFIED
# =========================
def hybrid_recommend(user_id, product_id, top_n=5, alpha=0.7):
    """
    Hybrid recommendation system combining CF and CB.
    
    Args:
        user_id: Target user ID
        product_id: Reference product ID
        top_n: Number of recommendations
        alpha: CF weight (0-1), CB weight = 1-alpha
    
    Returns:
        list: Recommended products with details
    """
    user_id = str(user_id).strip()
    product_id = str(product_id).strip()

    # Get scores from both algorithms
    cf_scores = get_cf_scores(user_id)
    cb_scores = get_cb_scores(product_id)
    
    # If no scores from either algorithm, return empty
    if cf_scores.empty and cb_scores.empty:
        return []
    
    # Normalize CF scores to [0, 1]
    if not cf_scores.empty:
        cf_min, cf_max = cf_scores.min(), cf_scores.max()
        if cf_max > cf_min:
            cf_norm = (cf_scores - cf_min) / (cf_max - cf_min)
        else:
            cf_norm = cf_scores / (cf_scores.max() + 1e-10)
    else:
        cf_norm = pd.Series(dtype=float)
    
    # Normalize CB scores to [0, 1]
    if not cb_scores.empty:
        cb_min, cb_max = cb_scores.min(), cb_scores.max()
        if cb_max > cb_min:
            cb_norm = (cb_scores - cb_min) / (cb_max - cb_min)
        else:
            cb_norm = cb_scores / (cb_scores.max() + 1e-10)
    else:
        cb_norm = pd.Series(dtype=float)
    
    # Combine scores
    if not cf_norm.empty and not cb_norm.empty:
        # Both have scores - combine with weights
        final_scores = alpha * cf_norm.add((1 - alpha) * cb_norm, fill_value=0)
    elif not cf_norm.empty:
        # Only CF has scores
        final_scores = cf_norm
    else:
        # Only CB has scores
        final_scores = cb_norm
    
    if final_scores.empty:
        return []
    
    # Get top N products
    top_products = final_scores.nlargest(top_n).index.tolist()
    
    # Build recommendation list with product details
    try:
        results = []
        for product in top_products:
            matching = df[df['product_id'] == product]
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
    except:
        return []
