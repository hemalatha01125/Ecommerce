const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000/api";
const TOKEN_KEY = "ers_access_token";
const USER_KEY = "ers_user";

export function getStoredAuth() {
  const token = localStorage.getItem(TOKEN_KEY);
  const rawUser = localStorage.getItem(USER_KEY);
  return {
    token,
    user: rawUser ? JSON.parse(rawUser) : null,
  };
}

export function storeAuth({ token, user }) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export async function apiRequest(path, options = {}) {
  const { token } = getStoredAuth();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error?.message || "Request failed.");
    error.status = response.status;
    error.code = data.error?.code;
    if (response.status === 401) {
      clearAuth();
      window.dispatchEvent(new CustomEvent("auth:expired", { detail: error.message }));
    }
    throw error;
  }

  return data;
}

export const api = {
  register: (payload) => apiRequest("/auth/register", { method: "POST", body: JSON.stringify(payload) }),
  login: (payload) => apiRequest("/auth/login", { method: "POST", body: JSON.stringify(payload) }),
  logout: () => apiRequest("/auth/logout", { method: "POST" }),
  me: () => apiRequest("/auth/me"),
  products: (params = {}) => apiRequest(`/products?${new URLSearchParams(params)}`),
  categories: () => apiRequest("/products/categories"),
  product: (id) => apiRequest(`/products/${encodeURIComponent(id)}`),
  similar: (id) => apiRequest(`/products/${encodeURIComponent(id)}/similar`),
  personalized: (id) => apiRequest(`/recommendations/personalized?${new URLSearchParams({ product_id: id })}`),
  behavior: (productId, eventType, score) =>
    apiRequest("/behavior", {
      method: "POST",
      body: JSON.stringify({ product_id: productId, event_type: eventType, score }),
    }),
  like: (productId) =>
    apiRequest("/behavior", { method: "POST", body: JSON.stringify({ product_id: productId, event_type: "like" }) }),
  rate: (productId, score) =>
    apiRequest("/behavior", {
      method: "POST",
      body: JSON.stringify({ product_id: productId, event_type: "rating", score }),
    }),
  wishlist: () => apiRequest("/wishlist"),
  addWishlist: (productId) => apiRequest("/wishlist", { method: "POST", body: JSON.stringify({ product_id: productId }) }),
  removeWishlist: (productId) => apiRequest(`/wishlist/${encodeURIComponent(productId)}`, { method: "DELETE" }),
  cart: () => apiRequest("/cart"),
  addCart: (productId, quantity = 1) =>
    apiRequest("/cart", { method: "POST", body: JSON.stringify({ product_id: productId, quantity }) }),
  updateCart: (productId, quantity) =>
    apiRequest(`/cart/${encodeURIComponent(productId)}`, { method: "PATCH", body: JSON.stringify({ quantity }) }),
  removeCart: (productId) => apiRequest(`/cart/${encodeURIComponent(productId)}`, { method: "DELETE" }),
  checkout: () => apiRequest("/cart/checkout", { method: "POST" }),
  users: () => apiRequest("/admin/users"),
};
