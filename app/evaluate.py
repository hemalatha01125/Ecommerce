import numpy as np
from app.recommender import predict_rating
from app.preprocessing import load_data   # better approach

# Load dataset properly
df = load_data()
print("Evaluation started...")
def calculate_rmse():
    print("Inside RMSE function...")

    actual = []
    predicted = []

    for _, row in df.iterrows():
        user = row['user_id']
        product = row['product_id']
        real_rating = row['rating']

        pred_rating = predict_rating(user, product)

        actual.append(real_rating)
        predicted.append(pred_rating)

    actual = np.array(actual)
    predicted = np.array(predicted)

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return rmse


# 👇 ADD THIS (IMPORTANT)
if __name__ == "__main__":
    print("RMSE:", calculate_rmse())