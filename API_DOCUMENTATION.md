# E-Commerce Recommender API

Base URL: `http://127.0.0.1:5000/api`

JWT access tokens are returned by login/register and must be sent as:

```http
Authorization: Bearer <token>
```

Tokens expire according to `JWT_ACCESS_TOKEN_EXPIRES_MINUTES` and are revoked server-side on logout via `token_sessions`.

## Auth

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| POST | `/auth/register` | Public | Creates a user account with role `user` and returns a JWT. |
| POST | `/auth/login` | Public | Authenticates by username or email and returns a JWT. |
| POST | `/auth/logout` | Bearer | Revokes the current JWT session. |
| GET | `/auth/me` | Bearer | Returns the authenticated user profile. |

Register body:

```json
{
  "username": "alice",
  "email": "alice@example.com",
  "password": "StrongPass1"
}
```

Login body:

```json
{
  "username": "alice",
  "password": "StrongPass1"
}
```

## Products

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| GET | `/products?search=&category=&limit=24&offset=0` | Public | Lists catalog products. |
| GET | `/products/categories` | Public | Lists top-level categories. |
| GET | `/products/<product_id>` | Public | Returns product image, title, description, category, price, rating, and stock status. |
| GET | `/products/<product_id>/similar` | Bearer | Returns content-based similar products. |
| GET | `/recommendations/personalized?product_id=<id>&limit=6` | Bearer | Returns personalized hybrid recommendations. |

## Wishlist And Cart

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| GET | `/wishlist` | Bearer | Lists saved wishlist products. |
| POST | `/wishlist` | Bearer | Adds `{ "product_id": "..." }`. |
| DELETE | `/wishlist/<product_id>` | Bearer | Removes a wishlist product. |
| GET | `/cart` | Bearer | Lists cart items. |
| POST | `/cart` | Bearer | Adds `{ "product_id": "...", "quantity": 1 }`. |
| PATCH | `/cart/<product_id>` | Bearer | Updates `{ "quantity": 2 }`; zero removes the item. |
| DELETE | `/cart/<product_id>` | Bearer | Removes a cart product. |

## Admin

| Method | Endpoint | Auth | Description |
| --- | --- | --- | --- |
| GET | `/admin/users` | Admin Bearer | Lists users. |

To grant admin access in SQLite during development, update a trusted user:

```sql
UPDATE users SET role = 'admin' WHERE username = 'alice';
```

## Errors

Errors use a consistent JSON shape:

```json
{
  "error": {
    "code": "token_expired",
    "message": "Token has expired. Please log in again."
  }
}
```

Common status codes: `400` validation error, `401` missing/invalid/expired token, `403` insufficient role, `404` not found, `409` duplicate account.
