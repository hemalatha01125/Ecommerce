from datetime import datetime, timedelta, timezone
from functools import wraps
from uuid import uuid4

import jwt
from flask import Blueprint, current_app, g, jsonify, request

from .models import CartItem, Role, TokenSession, User, WishlistItem, db
from .product_service import (
    get_categories,
    get_personalized_recommendations,
    get_product,
    get_similar_products,
    list_products,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _utcnow():
    return datetime.now(timezone.utc)


def _json_error(message, status_code=400, code="bad_request"):
    response = jsonify({"error": {"code": code, "message": message}})
    response.status_code = status_code
    return response


def _get_bearer_token():
    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _generate_access_token(user):
    issued_at = _utcnow()
    expires_at = issued_at + timedelta(minutes=current_app.config["JWT_ACCESS_TOKEN_EXPIRES_MINUTES"])
    jti = uuid4().hex
    payload = {
        "sub": str(user.id),
        "jti": jti,
        "role": user.role,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": "ecommerce-recommender",
    }
    token = jwt.encode(payload, current_app.config["JWT_SECRET_KEY"], algorithm=current_app.config["JWT_ALGORITHM"])
    session = TokenSession(
        jti=jti,
        user_id=user.id,
        issued_at=issued_at.replace(tzinfo=None),
        expires_at=expires_at.replace(tzinfo=None),
        user_agent=request.headers.get("User-Agent", "")[:255],
        ip_address=request.headers.get("X-Forwarded-For", request.remote_addr or "")[:45],
    )
    db.session.add(session)
    db.session.commit()
    return token, expires_at


def jwt_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _get_bearer_token()
        if not token:
            return _json_error("Missing bearer token.", 401, "missing_token")

        try:
            payload = jwt.decode(
                token,
                current_app.config["JWT_SECRET_KEY"],
                algorithms=[current_app.config["JWT_ALGORITHM"]],
                issuer="ecommerce-recommender",
            )
        except jwt.ExpiredSignatureError:
            return _json_error("Token has expired. Please log in again.", 401, "token_expired")
        except jwt.InvalidTokenError:
            return _json_error("Invalid authentication token.", 401, "invalid_token")

        session = TokenSession.query.filter_by(jti=payload.get("jti")).first()
        if not session or session.is_revoked:
            return _json_error("Session is no longer active.", 401, "session_revoked")

        user = User.query.get(payload.get("sub"))
        if not user or not user.is_active_account:
            return _json_error("User account is unavailable.", 401, "inactive_user")

        g.current_user = user
        g.jwt_payload = payload
        g.token_session = session
        return fn(*args, **kwargs)

    return wrapper


def roles_required(*roles):
    def decorator(fn):
        @wraps(fn)
        @jwt_required
        def wrapper(*args, **kwargs):
            if g.current_user.role not in roles:
                return _json_error("You do not have permission to access this resource.", 403, "forbidden")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def _request_json():
    return request.get_json(silent=True) or {}


def _positive_int(value, default=1):
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


def _validate_password(password):
    if not password or len(password) < 8:
        return "Password must be at least 8 characters long."
    if password.lower() == password or password.upper() == password or not any(ch.isdigit() for ch in password):
        return "Password must include uppercase, lowercase, and numeric characters."
    return None


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@api_bp.route("/auth/register", methods=["POST"])
def register():
    data = _request_json()
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return _json_error("Username, email, and password are required.")
    if len(username) < 3:
        return _json_error("Username must be at least 3 characters long.")
    password_error = _validate_password(password)
    if password_error:
        return _json_error(password_error)
    if User.query.filter_by(username=username).first():
        return _json_error("Username already exists.", 409, "username_exists")
    if User.query.filter_by(email=email).first():
        return _json_error("Email is already registered.", 409, "email_exists")

    user = User(username=username, email=email, role=Role.USER)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    token, expires_at = _generate_access_token(user)
    return jsonify({"token": token, "expires_at": expires_at.isoformat(), "user": user.to_dict()}), 201


@api_bp.route("/auth/login", methods=["POST"])
def login():
    data = _request_json()
    username = str(data.get("username", "")).strip()
    password = data.get("password", "")

    if not username or not password:
        return _json_error("Username and password are required.")

    user = User.query.filter((User.username == username) | (User.email == username.lower())).first()
    if not user or not user.check_password(password):
        return _json_error("Invalid username/email or password.", 401, "invalid_credentials")
    if not user.is_active_account:
        return _json_error("User account is inactive.", 403, "inactive_user")

    token, expires_at = _generate_access_token(user)
    return jsonify({"token": token, "expires_at": expires_at.isoformat(), "user": user.to_dict()})


@api_bp.route("/auth/logout", methods=["POST"])
@jwt_required
def logout():
    g.token_session.revoked_at = _utcnow().replace(tzinfo=None)
    db.session.commit()
    return jsonify({"message": "Logged out successfully."})


@api_bp.route("/auth/me", methods=["GET"])
@jwt_required
def me():
    return jsonify({"user": g.current_user.to_dict()})


@api_bp.route("/admin/users", methods=["GET"])
@roles_required(Role.ADMIN)
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({"items": [user.to_dict() for user in users]})


@api_bp.route("/products", methods=["GET"])
def products():
    limit = min(max(request.args.get("limit", 24, type=int), 1), 100)
    offset = max(request.args.get("offset", 0, type=int), 0)
    return jsonify(
        list_products(
            search=request.args.get("search"),
            category=request.args.get("category"),
            limit=limit,
            offset=offset,
        )
    )


@api_bp.route("/products/categories", methods=["GET"])
def categories():
    return jsonify({"items": get_categories()})


@api_bp.route("/products/<product_id>", methods=["GET"])
def product_detail(product_id):
    product = get_product(product_id)
    if not product:
        return _json_error("Product not found.", 404, "not_found")
    return jsonify({"product": product})


@api_bp.route("/products/<product_id>/similar", methods=["GET"])
@jwt_required
def similar_products(product_id):
    if not get_product(product_id):
        return _json_error("Product not found.", 404, "not_found")
    top_n = min(max(request.args.get("limit", 6, type=int), 1), 20)
    return jsonify({"items": get_similar_products(product_id, top_n=top_n)})


@api_bp.route("/recommendations/personalized", methods=["GET"])
@jwt_required
def personalized_recommendations():
    product_id = request.args.get("product_id", "").strip()
    engine_user_id = request.args.get("engine_user_id", g.current_user.username).strip()
    if not product_id:
        return _json_error("product_id is required.")
    if not get_product(product_id):
        return _json_error("Product not found.", 404, "not_found")
    top_n = min(max(request.args.get("limit", 6, type=int), 1), 20)
    return jsonify({"items": get_personalized_recommendations(engine_user_id, product_id, top_n=top_n)})


def _items_for(model):
    rows = model.query.filter_by(user_id=g.current_user.id).order_by(model.created_at.desc()).all()
    items = []
    for row in rows:
        product = get_product(row.product_id)
        if product:
            item = {"id": row.id, "product": product}
            if hasattr(row, "quantity"):
                item["quantity"] = row.quantity
            items.append(item)
    return items


@api_bp.route("/wishlist", methods=["GET"])
@jwt_required
def get_wishlist():
    return jsonify({"items": _items_for(WishlistItem)})


@api_bp.route("/wishlist", methods=["POST"])
@jwt_required
def add_wishlist():
    product_id = str(_request_json().get("product_id", "")).strip()
    if not get_product(product_id):
        return _json_error("Product not found.", 404, "not_found")
    existing = WishlistItem.query.filter_by(user_id=g.current_user.id, product_id=product_id).first()
    if not existing:
        db.session.add(WishlistItem(user_id=g.current_user.id, product_id=product_id))
        db.session.commit()
    return jsonify({"items": _items_for(WishlistItem)}), 201


@api_bp.route("/wishlist/<product_id>", methods=["DELETE"])
@jwt_required
def remove_wishlist(product_id):
    WishlistItem.query.filter_by(user_id=g.current_user.id, product_id=product_id).delete()
    db.session.commit()
    return jsonify({"items": _items_for(WishlistItem)})


@api_bp.route("/cart", methods=["GET"])
@jwt_required
def get_cart():
    return jsonify({"items": _items_for(CartItem)})


@api_bp.route("/cart", methods=["POST"])
@jwt_required
def add_cart():
    data = _request_json()
    product_id = str(data.get("product_id", "")).strip()
    quantity = _positive_int(data.get("quantity", 1))
    if not get_product(product_id):
        return _json_error("Product not found.", 404, "not_found")

    item = CartItem.query.filter_by(user_id=g.current_user.id, product_id=product_id).first()
    if item:
        item.quantity += quantity
    else:
        db.session.add(CartItem(user_id=g.current_user.id, product_id=product_id, quantity=quantity))
    db.session.commit()
    return jsonify({"items": _items_for(CartItem)}), 201


@api_bp.route("/cart/<product_id>", methods=["PATCH"])
@jwt_required
def update_cart(product_id):
    try:
        quantity = int(_request_json().get("quantity", 1))
    except (TypeError, ValueError):
        return _json_error("Quantity must be an integer.")
    item = CartItem.query.filter_by(user_id=g.current_user.id, product_id=product_id).first()
    if not item:
        return _json_error("Cart item not found.", 404, "not_found")
    if quantity <= 0:
        db.session.delete(item)
    else:
        item.quantity = quantity
    db.session.commit()
    return jsonify({"items": _items_for(CartItem)})


@api_bp.route("/cart/<product_id>", methods=["DELETE"])
@jwt_required
def remove_cart(product_id):
    CartItem.query.filter_by(user_id=g.current_user.id, product_id=product_id).delete()
    db.session.commit()
    return jsonify({"items": _items_for(CartItem)})
