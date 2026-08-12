import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Heart, LogOut, Search, ShoppingCart, ShieldCheck, Star, ThumbsUp, UserRound } from "lucide-react";
import { AuthProvider, useAuth } from "./AuthContext";
import { api } from "./api";
import "./styles.css";

const FALLBACK_IMAGE = "http://127.0.0.1:5000/static/images/image-not-available.png";

function useHashRoute() {
  const [hash, setHash] = useState(window.location.hash || "#/");

  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const [path, queryString = ""] = hash.slice(1).split("?");
  return { path: path || "/", params: new URLSearchParams(queryString) };
}

function navigate(path) {
  window.location.hash = path;
}

function Protected({ children, adminOnly = false }) {
  const { token, user } = useAuth();
  if (!token) {
    return <AuthForm mode="login" />;
  }
  if (adminOnly && user?.role !== "admin") {
    return <StateMessage title="Unauthorized" message="This area is only available to administrators." />;
  }
  return children;
}

function Shell({ children }) {
  const { user, token, logout, notice, clearNotice } = useAuth();
  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/">
          <ShoppingCart size={22} /> Recommender Store
        </a>
        <nav>
          <a href="#/">Products</a>
          {token && <a href="#/wishlist">Wishlist</a>}
          {token && <a href="#/cart">Cart</a>}
          {user?.role === "admin" && <a href="#/admin">Admin</a>}
        </nav>
        <div className="account">
          {user ? (
            <>
              <span>
                <UserRound size={16} /> {user.username}
              </span>
              <button className="icon-button" onClick={logout} title="Log out">
                <LogOut size={18} />
              </button>
            </>
          ) : (
            <>
              <button onClick={() => navigate("/login")}>Login</button>
              <button className="primary" onClick={() => navigate("/register")}>
                Register
              </button>
            </>
          )}
        </div>
      </header>
      {notice && (
        <button className="notice" onClick={clearNotice}>
          {notice}
        </button>
      )}
      <main className="page">{children}</main>
    </>
  );
}

function AuthForm({ mode }) {
  const auth = useAuth();
  const isRegister = mode === "register";
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (isRegister) {
        await auth.register(form);
      } else {
        await auth.login({ username: form.username, password: form.password });
      }
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="auth-panel">
      <h1>{isRegister ? "Create Account" : "Welcome Back"}</h1>
      <form onSubmit={submit}>
        <label>
          Username or email
          <input value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} />
        </label>
        {isRegister && (
          <label>
            Email
            <input type="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
          </label>
        )}
        <label>
          Password
          <input
            type="password"
            value={form.password}
            onChange={(event) => setForm({ ...form, password: event.target.value })}
          />
        </label>
        {error && <p className="error">{error}</p>}
        <button className="primary wide" disabled={loading}>
          {loading ? "Please wait..." : isRegister ? "Register" : "Login"}
        </button>
      </form>
      <button className="link-button" onClick={() => navigate(isRegister ? "/login" : "/register")}>
        {isRegister ? "Already have an account?" : "Need an account?"}
      </button>
    </section>
  );
}

function Catalog() {
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api.categories().then((data) => setCategories(data.items)).catch(() => setCategories([]));
  }, []);

  useEffect(() => {
    setLoading(true);
    api
      .products({ search, category, limit: 24 })
      .then((data) => {
        setProducts(data.items);
        setError("");
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [search, category]);

  return (
    <>
      <section className="toolbar">
        <div className="searchbox">
          <Search size={18} />
          <input placeholder="Search products" value={search} onChange={(event) => setSearch(event.target.value)} />
        </div>
        <select value={category} onChange={(event) => setCategory(event.target.value)}>
          <option value="">All categories</option>
          {categories.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </section>
      {loading && <StateMessage title="Loading products" />}
      {error && <StateMessage title="Could not load products" message={error} />}
      <section className="product-grid">
        {products.map((product) => (
          <ProductCard key={product.product_id} product={product} />
        ))}
      </section>
    </>
  );
}

function ProductCard({ product }) {
  return (
    <article className="product-card" onClick={() => navigate(`/product/${product.product_id}`)}>
      <img
        src={product.image || FALLBACK_IMAGE}
        alt={product.title}
        loading="lazy"
        onError={(event) => {
          if (event.currentTarget.src === FALLBACK_IMAGE) return;
          event.currentTarget.src = FALLBACK_IMAGE;
        }}
      />
      <div>
        <h2>{product.title}</h2>
        <p>{product.category?.split("|").slice(0, 2).join(" / ")}</p>
        <strong>{product.price || "Price unavailable"}</strong>
        <span>{product.rating ? `${product.rating.toFixed(1)} rating` : "No rating"}</span>
      </div>
    </article>
  );
}

function ProductDetail({ productId }) {
  const { token } = useAuth();
  const [product, setProduct] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [personalized, setPersonalized] = useState([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .product(productId)
      .then((data) => setProduct(data.product))
      .catch((err) => setError(err.message));
  }, [productId]);

  useEffect(() => {
    if (!token) return;
    api.similar(productId).then((data) => setSimilar(data.items)).catch((err) => setMessage(err.message));
    api.personalized(productId).then((data) => setPersonalized(data.items)).catch(() => setPersonalized([]));
  }, [productId, token]);

  async function action(task) {
    try {
      await task(productId);
      setMessage("Saved.");
      api.personalized(productId).then((data) => setPersonalized(data.items)).catch(() => setPersonalized([]));
    } catch (err) {
      setMessage(err.message);
    }
  }

  if (error) return <StateMessage title="Product unavailable" message={error} />;
  if (!product) return <StateMessage title="Loading product" />;

  return (
    <>
      <section className="detail">
        <img
          src={product.image || FALLBACK_IMAGE}
          alt={product.title}
          onError={(event) => {
            if (event.currentTarget.src === FALLBACK_IMAGE) return;
            event.currentTarget.src = FALLBACK_IMAGE;
          }}
        />
        <div>
          <p className="category">{product.category}</p>
          <h1>{product.title}</h1>
          <p className="description">{product.description}</p>
          <div className="facts">
            <strong>{product.price}</strong>
            <span>{product.rating ? `${product.rating.toFixed(1)} rating` : "No rating"}</span>
            <span className={product.stock_status === "In stock" ? "stock in" : "stock out"}>{product.stock_status}</span>
          </div>
          {token ? (
            <>
              <div className="actions">
                <button className="primary" onClick={() => action(api.addCart)}>
                  <ShoppingCart size={18} /> Add to cart
                </button>
                <button onClick={() => action(api.addWishlist)}>
                  <Heart size={18} /> Wishlist
                </button>
                <button onClick={() => action(api.like)}>
                  <ThumbsUp size={18} /> Like
                </button>
              </div>
              <div className="rating-actions" aria-label="Rate product">
                {[1, 2, 3, 4, 5].map((score) => (
                  <button key={score} onClick={() => action((id) => api.rate(id, score))} title={`Rate ${score}`}>
                    <Star size={16} /> {score}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <p className="callout">Log in to view recommendations, save wishlist items, and use your cart.</p>
          )}
          {message && <p className="muted">{message}</p>}
        </div>
      </section>
      {token && (
        <>
          <ProductRail title="Recommended Similar Products" items={similar} />
          <ProductRail title="Personalized Recommendations" items={personalized} />
        </>
      )}
    </>
  );
}

function ProductRail({ title, items }) {
  if (!items.length) return null;
  return (
    <section>
      <h2 className="section-title">{title}</h2>
      <div className="product-grid compact">
        {items.map((product) => (
          <ProductCard key={product.product_id} product={product} />
        ))}
      </div>
    </section>
  );
}

function SavedItems({ type }) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");
  const load = () => api[type]().then((data) => setItems(data.items)).catch((err) => setError(err.message));

  useEffect(() => {
    load();
  }, [type]);

  async function remove(productId) {
    if (type === "cart") {
      await api.removeCart(productId);
    } else {
      await api.removeWishlist(productId);
    }
    load();
  }

  async function checkout() {
    try {
      await api.checkout();
      setError("");
      load();
    } catch (err) {
      setError(err.message);
    }
  }

  if (error) return <StateMessage title={`Could not load ${type}`} message={error} />;
  return (
    <section>
      <h1 className="section-title">{type === "cart" ? "Cart" : "Wishlist"}</h1>
      {type === "cart" && items.length > 0 && (
        <button className="primary checkout-button" onClick={checkout}>
          <ShoppingCart size={18} /> Checkout
        </button>
      )}
      <div className="saved-list">
        {items.map(({ product, quantity }) => (
          <article key={product.product_id}>
            <img
              src={product.image || FALLBACK_IMAGE}
              alt={product.title}
              onError={(event) => {
                if (event.currentTarget.src === FALLBACK_IMAGE) return;
                event.currentTarget.src = FALLBACK_IMAGE;
              }}
            />
            <div>
              <h2>{product.title}</h2>
              <p>{product.price} {quantity ? `x ${quantity}` : ""}</p>
            </div>
            <button onClick={() => remove(product.product_id)}>Remove</button>
          </article>
        ))}
      </div>
    </section>
  );
}

function AdminUsers() {
  const [users, setUsers] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api.users().then((data) => setUsers(data.items)).catch((err) => setError(err.message));
  }, []);

  if (error) return <StateMessage title="Admin request failed" message={error} />;
  return (
    <section>
      <h1 className="section-title">
        <ShieldCheck size={24} /> Users
      </h1>
      <div className="table">
        {users.map((user) => (
          <div key={user.id}>
            <span>{user.username}</span>
            <span>{user.email}</span>
            <strong>{user.role}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function StateMessage({ title, message = "" }) {
  return (
    <section className="state">
      <h1>{title}</h1>
      {message && <p>{message}</p>}
    </section>
  );
}

function AppRoutes() {
  const route = useHashRoute();
  const productMatch = useMemo(() => route.path.match(/^\/product\/(.+)$/), [route.path]);

  if (route.path === "/login") return <AuthForm mode="login" />;
  if (route.path === "/register") return <AuthForm mode="register" />;
  if (route.path === "/wishlist") return <Protected><SavedItems type="wishlist" /></Protected>;
  if (route.path === "/cart") return <Protected><SavedItems type="cart" /></Protected>;
  if (route.path === "/admin") return <Protected adminOnly><AdminUsers /></Protected>;
  if (productMatch) return <ProductDetail productId={decodeURIComponent(productMatch[1])} />;
  return <Catalog />;
}

function App() {
  return (
    <AuthProvider>
      <Shell>
        <AppRoutes />
      </Shell>
    </AuthProvider>
  );
}

createRoot(document.getElementById("root")).render(<App />);
