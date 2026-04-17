from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from .recommender import hybrid_recommend

main = Blueprint('main', __name__)

@main.route('/')
def home():
    """Home page - redirect to dashboard if logged in."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    """User dashboard for recommendations."""
    return render_template('dashboard.html', username=current_user.username)

@main.route('/recommend', methods=['POST'])
@login_required
def recommend():
    """Get product recommendations - requires login."""
    user_id = request.form['user_id'].strip()
    product_id = request.form['product_id'].strip()
    
    if not user_id or not product_id:
        flash('Please enter both User ID and Product ID.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    current_app.logger.info(
        "Recommend request received from user %s: user_id=%s, product_id=%s",
        current_user.username,
        user_id,
        product_id,
    )
    
    try:
        results = hybrid_recommend(user_id, product_id)
        current_app.logger.info("Sending %s recommendations to template", len(results))
        current_app.logger.debug("Recommendation payload: %s", results)
        return render_template('recommendations.html', results=results, user_id=user_id, product_id=product_id)
    except Exception as e:
        current_app.logger.error(f"Error getting recommendations: {e}")
        flash('Error getting recommendations. Please try again.', 'danger')
        return redirect(url_for('main.dashboard'))

