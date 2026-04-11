import os

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
    
    from .routes import main
    app.register_blueprint(main)
    
    return app
