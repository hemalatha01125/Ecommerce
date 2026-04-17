import os
from flask import Flask
from flask_login import LoginManager
from .models import db, User


def create_app():
    try:
        from flask import Flask
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Flask is not installed. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    app = Flask(
        __name__,
        template_folder=os.path.join(os.getcwd(), "templates"),
        static_folder=os.path.join(os.getcwd(), "static")
    )
    
    # Load configuration
    from config import config_by_name
    config_mode = os.environ.get('FLASK_ENV', 'development')
    app.config.from_object(config_by_name.get(config_mode, config_by_name['development']))
    
    # Initialize database
    db.init_app(app)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Register blueprints
    from .routes import main
    from .auth import auth_bp
    app.register_blueprint(main)
    app.register_blueprint(auth_bp)
    
    return app

