import os
from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager
from .models import db, User


def _ensure_user_columns(app):
    """Add new auth columns when running against an older SQLite database."""
    if not app.config.get("SQLALCHEMY_DATABASE_URI", "").startswith("sqlite"):
        return

    with db.engine.connect() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(users)").fetchall()}
        if "role" not in columns:
            connection.exec_driver_sql("ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'")
        if "is_active_account" not in columns:
            connection.exec_driver_sql("ALTER TABLE users ADD COLUMN is_active_account BOOLEAN NOT NULL DEFAULT 1")
        connection.commit()


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

    cors_origins = [
        origin.strip()
        for origin in app.config.get("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    CORS(app, resources={r"/api/*": {"origins": cors_origins}}, supports_credentials=False)
    
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
        _ensure_user_columns(app)
    
    # Register blueprints
    from .routes import main
    from .auth import auth_bp
    from .api import api_bp
    app.register_blueprint(main)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)
    
    return app

