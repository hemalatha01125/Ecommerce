import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/amazon.csv")

# Keep required columns
df = df[['user_id', 'product_id', 'product_name', 'category', 'about_product', 'rating', 'discounted_price']]

# Drop missing values
df = df.dropna()

# Convert rating
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])
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
def get_cf_scores(user_id):
    if user_id not in user_item.index:
        return pd.Series()

    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]

    scores = pd.Series(dtype=float)

    for sim_user, similarity in similar_users.items():
        user_ratings = user_item.loc[sim_user]
        scores = scores.add(user_ratings * similarity, fill_value=0)

    # Remove already rated
    rated = user_item.loc[user_id].dropna().index
    scores = scores.drop(rated, errors='ignore')

    return scores


# =========================
# CB FUNCTION
# =========================
def get_cb_scores(product_id):
    if product_id not in df['product_id'].values:
        return pd.Series()

    idx = df[df['product_id'] == product_id].index[0]

    sim_scores = pd.Series(content_sim[idx], index=df['product_id'])

    return sim_scores


# =========================
# HYBRID RECOMMENDER
# =========================
def hybrid_recommend(user_id, product_id, top_n=5, alpha=0.7):

    # Get scores
    cf_scores = get_cf_scores(user_id)
    cb_scores = get_cb_scores(product_id)

    # Normalize
    if not cf_scores.empty:
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)

    if not cb_scores.empty:
        cb_scores = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min() + 1e-9)

    # Combine
    final_scores = alpha * cf_scores.add(cb_scores, fill_value=0)

    # Sort
    final_scores = final_scores.sort_values(ascending=False)

    # Get top products
    top_products = final_scores.head(top_n).index

    # Return full details
    results = df[df['product_id'].isin(top_products)][
        ['product_id', 'product_name', 'rating', 'discounted_price']
    ]

    return results