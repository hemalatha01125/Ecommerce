import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api, clearAuth, getStoredAuth, storeAuth } from "./api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(getStoredAuth);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const onExpired = (event) => {
      setAuth({ token: null, user: null });
      setNotice(event.detail || "Your session expired. Please log in again.");
    };
    window.addEventListener("auth:expired", onExpired);
    return () => window.removeEventListener("auth:expired", onExpired);
  }, []);

  const value = useMemo(
    () => ({
      ...auth,
      notice,
      clearNotice: () => setNotice(""),
      async login(payload) {
        const result = await api.login(payload);
        storeAuth(result);
        setAuth({ token: result.token, user: result.user });
      },
      async register(payload) {
        const result = await api.register(payload);
        storeAuth(result);
        setAuth({ token: result.token, user: result.user });
      },
      async logout() {
        try {
          if (auth.token) {
            await api.logout();
          }
        } finally {
          clearAuth();
          setAuth({ token: null, user: null });
        }
      },
    }),
    [auth, notice],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
