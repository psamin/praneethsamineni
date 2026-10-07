// Two pages don't justify a router dependency. Paths: "/" and "/projects".
// GitHub Pages serves 404.html (a copy of index.html) for /projects on reload.
import { useEffect, useState } from "react";

const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");

export const href = (path: string) => `${BASE}${path}`;

function current() {
  const p = window.location.pathname.slice(BASE.length) || "/";
  return p.replace(/\/+$/, "") || "/";
}

export function usePath() {
  const [path, setPath] = useState(current);
  useEffect(() => {
    const on = () => setPath(current());
    window.addEventListener("popstate", on);
    return () => window.removeEventListener("popstate", on);
  }, []);
  return path;
}

/** Client-side navigation for plain <a> clicks (keeps ctrl/cmd-click working). */
export function navigate(e: React.MouseEvent<HTMLAnchorElement>) {
  if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  const url = new URL(e.currentTarget.href);
  if (url.origin !== window.location.origin) return;
  e.preventDefault();
  if (url.pathname !== window.location.pathname) {
    // the page switches first; App scrolls to the hash once it has rendered
    window.history.pushState(null, "", url.pathname + url.hash);
    window.dispatchEvent(new PopStateEvent("popstate"));
  } else if (url.hash) {
    window.history.pushState(null, "", url.hash);
    document.querySelector(url.hash)?.scrollIntoView();
  } else {
    window.scrollTo(0, 0);
  }
}
