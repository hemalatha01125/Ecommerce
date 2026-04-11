from flask import Blueprint, current_app, render_template, request
from .recommender import hybrid_recommend

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form['user_id'].strip()
    product_id = request.form['product_id'].strip()
    
    current_app.logger.info(
        "Recommend request received: user_id=%s, product_id=%s",
        user_id,
        product_id,
    )
    results = hybrid_recommend(user_id, product_id)
    current_app.logger.info("Sending %s recommendations to template", len(results))
    current_app.logger.debug("Recommendation payload: %s", results)
    return render_template('recommendations.html', results=results)
