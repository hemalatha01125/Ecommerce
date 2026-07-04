# E-Commerce Recommender System

A hybrid recommendation system that combines collaborative filtering and content-based filtering to provide personalized product suggestions.

## Features

- Hybrid recommendation model using collaborative filtering and TF-IDF content similarity
- Flask backend with SQLite by default
- JWT authentication for API clients
- User registration, login, logout, password hashing, token validation, and token revocation
- Role-based authorization with `admin` and `user` roles
- Protected recommendation, wishlist, and cart APIs
- React frontend with protected routes, product catalog, product detail page, wishlist, and cart
- Product detail data includes image, title, description, category, price, rating, stock status, and recommended similar products

## Project Structure

```text
app/
  api.py              JWT API routes and authorization middleware
  auth.py             Existing Flask-Login template auth
  models.py           SQLAlchemy users, sessions, wishlist, and cart models
  product_service.py  Product catalog helpers for API responses
  recommender.py      Recommendation engine
frontend/
  src/                React app
templates/            Existing server-rendered Flask pages
static/               Existing Flask static assets
data/raw/amazon.csv   Product and interaction data
run.py                Flask entry point
API_DOCUMENTATION.md  REST API reference
```

## Backend Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Backend URL:

```text
http://127.0.0.1:5000
```

Useful environment variables:

```powershell
$env:SECRET_KEY="replace-me"
$env:JWT_SECRET_KEY="replace-me-too"
$env:JWT_ACCESS_TOKEN_EXPIRES_MINUTES="60"
$env:CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

React URL:

```text
http://127.0.0.1:5173
```

Set a custom backend URL with:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:5000/api"
```

## API Documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Admin Role

Public registration always creates a `user` account. For local development, promote a trusted account in SQLite:

```sql
UPDATE users SET role = 'admin' WHERE username = 'alice';
```

## Notes

- SQLite is used by default through `sqlite:///recommender.db`; set `DATABASE_URL` to use another database such as MySQL.
- Existing Flask template pages still work with Flask-Login.
- The source dataset has no live inventory column, so product stock status is a stable derived value for UI demonstration.
