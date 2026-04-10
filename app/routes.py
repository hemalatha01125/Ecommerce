from flask import Blueprint, render_template, request
from .recommender import hybrid_recommend

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form['user_id']
    product_id = request.form['product_id']
    
    results = hybrid_recommend(user_id, product_id).to_dict(orient='records')
    return render_template('recommendations.html', results=results)