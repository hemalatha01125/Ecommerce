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
        return pd.Series(dtype=float)

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
        return pd.Series(dtype=float)

    idx = df[df['product_id'] == product_id].index[0]

    sim_scores = pd.Series(content_sim[idx], index=df['product_id'])

    return sim_scores


# =========================
# HYBRID RECOMMENDER
# =========================
def hybrid_recommend(user_id, product_id, top_n=5, alpha=0.7):
    user_id = str(user_id).strip()
    product_id = str(product_id).strip()

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
    top_products = final_scores.head(top_n).index.tolist()

    if not top_products:
        print(f"[hybrid_recommend] No recommendations found for user_id={user_id}, product_id={product_id}")
        return []

    # Return full details in recommendation order
    results = (
        df[df['product_id'].isin(top_products)][
            ['product_id', 'product_name', 'rating', 'discounted_price', 'img_link']
        ]
        .drop_duplicates(subset=['product_id'])
        .set_index('product_id')
        .reindex(top_products)
        .reset_index()
        .fillna('')
        .to_dict(orient='records')
    )
    print(results)

    return results
