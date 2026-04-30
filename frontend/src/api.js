import { useEffect, useState } from 'react';

// Vite dev server proxies `/api/*` to http://127.0.0.1:8000 (see vite.config.js),
// so the same code works in dev and when served from the FastAPI host in prod.
const BASE = '/api';

async function getJson(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

export function imageUrl(name) {
  return `${BASE}/figures/${name}`;
}

// Fetch the three JSON bundles the dashboard needs in one hook. Returns
// { data, error, loading } — consumers render a skeleton while loading is
// true, then use data.metrics / data.ablation / data.features directly.
export function useApiData() {
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    let cancelled = false;
    Promise.all([getJson('/metrics'), getJson('/ablation'), getJson('/features')])
      .then(([metrics, ablation, features]) => {
        if (cancelled) return;
        setState({ data: { metrics, ablation, features }, error: null, loading: false });
      })
      .catch((err) => {
        if (cancelled) return;
        setState({ data: null, error: err.message, loading: false });
      });
    return () => { cancelled = true; };
  }, []);

  return state;
}

export async function predict(features) {
  const r = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  });
  if (!r.ok) throw new Error(`predict: ${r.status}`);
  return r.json();
}

export async function fetchSample() {
  return getJson('/sample');
}
