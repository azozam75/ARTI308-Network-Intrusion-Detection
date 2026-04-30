import { useCallback, useEffect, useMemo, useState } from 'react';

import { fmtFull, mkEv, rF, rInt } from './constants';
import { useApiData } from './api';
import {
  DashboardView, EventsView, ModelView, NetworkView, ReportsView,
} from './views';

const NAV = [
  { id: 'dashboard', icon: '⬡', label: 'Dashboard' },
  { id: 'model', icon: '◈', label: 'Model' },
  { id: 'events', icon: '◫', label: 'Events' },
  { id: 'network', icon: '◎', label: 'Network' },
  { id: 'reports', icon: '▤', label: 'Reports' },
];

const TWEAK_DEFAULTS = {
  density: 'compact',
  liveUpdates: true,
  showSparklines: true,
};

const Sidebar = ({ active, onNav }) => (
  <div style={{
    width: 68, background: 'var(--surface)', borderRight: '1px solid var(--border)',
    display: 'flex', flexDirection: 'column', alignItems: 'stretch', padding: '10px 6px', gap: 2, flexShrink: 0,
  }}>
    {NAV.map((item) => {
      const isActive = active === item.id;
      return (
        <button key={item.id} onClick={() => onNav(item.id)} title={item.label} style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 3, padding: '8px 4px', borderRadius: 4, cursor: 'pointer',
          background: isActive ? 'var(--amber-bg)' : 'transparent',
          border: isActive ? '1px solid var(--amber)55' : '1px solid transparent',
          color: isActive ? 'var(--amber)' : 'var(--dim)',
        }}>
          <span style={{ fontSize: 18, lineHeight: 1 }}>{item.icon}</span>
          <span style={{
            font: '500 9px var(--sans)', letterSpacing: '0.08em', textTransform: 'uppercase',
            color: isActive ? 'var(--amber)' : 'var(--dim)',
          }}>{item.label}</span>
        </button>
      );
    })}
  </div>
);

const TopBar = ({ clock, liveUpdates, config, tweaksOpen, onToggleTweaks }) => (
  <div style={{
    display: 'flex', alignItems: 'center', height: 48, background: 'var(--surface)',
    borderBottom: '1px solid var(--border)', padding: '0 18px', flexShrink: 0, gap: 0,
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginRight: 32 }}>
      <svg width="20" height="20" viewBox="0 0 20 20">
        <polygon points="10,1 19,6 19,14 10,19 1,14 1,6" fill="none" stroke="var(--amber)" strokeWidth="1.5" />
        <polygon points="10,5 15,8 15,12 10,15 5,12 5,8" fill="var(--amber)" opacity="0.18" />
        <circle cx="10" cy="10" r="2.2" fill="var(--amber)" />
      </svg>
      <span style={{ font: '700 14px var(--mono)', letterSpacing: '0.18em', color: 'var(--amber)' }}>SENTINEL</span>
      <span style={{ font: '400 11px var(--mono)', color: 'var(--dim)', letterSpacing: '0.06em' }}>Isolation Forest · CIC-IDS-2017</span>
    </div>
    <div style={{ display: 'flex', gap: 24, flex: 1 }}>
      {[
        ['n_estimators', String(config?.n_estimators ?? 200)],
        ['n_features', '50'],
        ['contamination', String(config?.contamination ?? 'auto')],
        ['split', '70/15/15'],
      ].map(([k, v]) => (
        <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ font: '500 9px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--dim)' }}>{k}</span>
          <span style={{ font: '600 11px var(--mono)', color: 'var(--muted)' }}>{v}</span>
        </div>
      ))}
    </div>
    <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
      {liveUpdates && (
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ animation: 'blink 1s steps(1) infinite', fontSize: 9, color: 'var(--green)' }}>●</span>
          <span style={{ font: '500 11px var(--mono)', color: 'var(--green)', letterSpacing: '0.08em' }}>STREAMING</span>
        </span>
      )}
      <span style={{ font: '500 12px var(--mono)', color: 'var(--muted)' }}>{clock}</span>
      <button onClick={onToggleTweaks} title="Toggle Tweaks panel" style={{
        display: 'flex', alignItems: 'center', gap: 6, padding: '6px 10px', borderRadius: 4, cursor: 'pointer',
        background: tweaksOpen ? 'var(--amber-bg)' : 'transparent',
        border: `1px solid ${tweaksOpen ? 'var(--amber)55' : 'var(--border)'}`,
        color: tweaksOpen ? 'var(--amber)' : 'var(--muted)',
        font: '600 10px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase',
      }}>
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4">
          <circle cx="6" cy="6" r="1.6" />
          <path d="M6 1v1.5M6 9.5V11M1 6h1.5M9.5 6H11M2.5 2.5l1 1M8.5 8.5l1 1M9.5 2.5l-1 1M3.5 8.5l-1 1" />
        </svg>
        Tweaks
      </button>
      <div style={{
        width: 30, height: 30, borderRadius: '50%', background: 'var(--surface2)', border: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', font: '600 11px var(--sans)', color: 'var(--amber)', cursor: 'pointer',
      }}>AO</div>
    </div>
  </div>
);

const TweaksPanel = ({ visible, tweaks, onChange, threshold, onThresholdChange }) => {
  if (!visible) return null;
  return (
    <div style={{
      position: 'fixed', bottom: 16, right: 16, zIndex: 999, background: 'var(--surface)',
      border: '1px solid var(--border)', borderRadius: 4, width: 240, padding: 16,
      boxShadow: '0 8px 32px oklch(0.05 0.02 245/0.9)',
    }}>
      <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--amber)', marginBottom: 14 }}>Tweaks</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div>
          <div style={{ font: '500 10px var(--sans)', color: 'var(--muted)', marginBottom: 6, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Row Density</div>
          <div style={{ display: 'flex', gap: 4 }}>
            {['compact', 'comfortable', 'spacious'].map((d) => (
              <button key={d} onClick={() => onChange('density', d)} style={{
                flex: 1, font: '500 10px var(--mono)', padding: '5px 0', borderRadius: 3, cursor: 'pointer',
                background: tweaks.density === d ? 'var(--amber-bg)' : 'var(--surface2)',
                border: `1px solid ${tweaks.density === d ? 'var(--amber)55' : 'var(--border)'}`,
                color: tweaks.density === d ? 'var(--amber)' : 'var(--muted)',
              }}>{d.slice(0, 4)}</button>
            ))}
          </div>
        </div>
        <div>
          <div style={{ font: '500 10px var(--sans)', color: 'var(--muted)', marginBottom: 6, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Live Event Stream</div>
          <button onClick={() => onChange('liveUpdates', !tweaks.liveUpdates)} style={{
            width: '100%', font: '500 11px var(--mono)', padding: '6px 0', borderRadius: 3, cursor: 'pointer',
            background: tweaks.liveUpdates ? 'var(--green-bg)' : 'var(--surface2)',
            border: `1px solid ${tweaks.liveUpdates ? 'var(--green)55' : 'var(--border)'}`,
            color: tweaks.liveUpdates ? 'var(--green)' : 'var(--muted)',
          }}>{tweaks.liveUpdates ? '● STREAMING' : '○ PAUSED'}</button>
        </div>
        <div>
          <div style={{ font: '500 10px var(--sans)', color: 'var(--muted)', marginBottom: 4, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            Decision Threshold: <span style={{ color: 'var(--amber)', fontFamily: 'var(--mono)' }}>{threshold.toFixed(4)}</span>
          </div>
          <input type="range" min="-0.20" max="0.15" step="0.001"
            value={threshold} onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--amber)' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', font: '400 9px var(--mono)', color: 'var(--dim)', marginTop: 2 }}>
            <span>sensitive</span><span>strict</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const LoadingOverlay = ({ error }) => (
  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 12 }}>
    {error ? (
      <>
        <span style={{ font: '700 14px var(--mono)', color: 'var(--red)' }}>Backend unreachable</span>
        <span style={{ font: '400 12px var(--mono)', color: 'var(--muted)', maxWidth: 420, textAlign: 'center' }}>{error}</span>
        <span style={{ font: '400 11px var(--mono)', color: 'var(--dim)' }}>Start the API: <code>python -m uvicorn backend.main:app --reload</code></span>
      </>
    ) : (
      <>
        <span style={{ font: '700 14px var(--mono)', color: 'var(--amber)', letterSpacing: '0.15em' }}>LOADING</span>
        <span style={{ font: '400 11px var(--mono)', color: 'var(--muted)' }}>fetching metrics, ablation, features…</span>
      </>
    )}
  </div>
);

export default function App() {
  const { data, error, loading } = useApiData();
  const [tuningResults, setTuningResults] = useState(null);
  const [tweaks, setTweaks] = useState(() => {
    try { return { ...TWEAK_DEFAULTS, ...JSON.parse(localStorage.getItem('sentinel-nids') || '{}') }; }
    catch { return TWEAK_DEFAULTS; }
  });
  // Derived threshold: user slider overrides, otherwise falls through to the
  // API-fetched value. Avoids the "setState in effect" cascade that a
  // naive `setThreshold(data.metrics.threshold)` on arrival would trigger.
  const [userThreshold, setUserThreshold] = useState(null);
  const threshold = userThreshold ?? data?.metrics.threshold ?? -0.0397;
  const [tweaksOpen, setTweaksOpen] = useState(false);
  const [nav, setNav] = useState('dashboard');
  const [events, setEvents] = useState([]);
  const [clock, setClock] = useState(fmtFull(new Date()));
  const [sparkEPS, setSparkEPS] = useState(() => Array.from({ length: 22 }, () => rInt(600, 2800)));
  const [sparkDetect, setSparkDetect] = useState(() => Array.from({ length: 22 }, () => rF(0.50, 0.60)));
  const [sparkScore, setSparkScore] = useState(() => Array.from({ length: 22 }, () => rF(-0.16, -0.10)));

  const attackRecall = useMemo(() => {
    if (!data) return null;
    const out = {};
    Object.entries(data.metrics.val_attack_metrics).forEach(([cat, v]) => {
      out[cat] = {
        recall: v.recall,
        false_positive_rate: v.false_positive_rate,
        n: v.n,
        score_mean: v.score_mean,
        score_std: v.score_std || 0.03,
      };
    });
    return out;
  }, [data]);

  const modelMetrics = data?.metrics.val_metrics ?? null;

  useEffect(() => {
    fetch('/api/tuning').then((r) => r.ok ? r.json() : null).then(setTuningResults).catch(() => {});
  }, []);

  useEffect(() => {
    localStorage.setItem('sentinel-nids', JSON.stringify(tweaks));
  }, [tweaks]);

  useEffect(() => {
    const id = setInterval(() => setClock(fmtFull(new Date())), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!attackRecall) return;
    const init = Array.from({ length: 100 }, () => {
      const e = mkEv(attackRecall, threshold);
      e.ts = new Date(Date.now() - rInt(0, 3600000));
      return e;
    }).sort((a, b) => b.ts - a.ts);
    ['DDoS', 'DoS', 'DoS', 'DDoS', 'DoS'].forEach((c, i) => {
      if (!init[i]) return;
      init[i].cat = c;
      init[i].score = parseFloat(rF(0.05, 0.14).toFixed(4));
      init[i].flagged = true;
      init[i].status = 'DETECTED';
    });
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setEvents(init);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attackRecall]);

  useEffect(() => {
    if (!tweaks.liveUpdates || !attackRecall) return;
    const id = setInterval(() => {
      setEvents((p) => [mkEv(attackRecall, threshold), ...p].slice(0, 250));
      setSparkEPS((p) => [...p.slice(1), rInt(600, 2800)]);
      setSparkDetect((p) => [...p.slice(1), rF(0.50, 0.60)]);
      setSparkScore((p) => [...p.slice(1), rF(-0.16, -0.08)]);
    }, 2800);
    return () => clearInterval(id);
  }, [tweaks.liveUpdates, threshold, attackRecall]);

  const dismissEv = useCallback((id) => setEvents((p) => p.filter((e) => e.id !== id)), []);
  const setTweak = (k, v) => setTweaks((prev) => ({ ...prev, [k]: v }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <TopBar clock={clock} liveUpdates={tweaks.liveUpdates} config={data?.metrics.config}
        tweaksOpen={tweaksOpen} onToggleTweaks={() => setTweaksOpen((v) => !v)} />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar active={nav} onNav={setNav} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {(loading || error || !data) ? (
            <LoadingOverlay error={error} />
          ) : (
            <>
              {nav === 'dashboard' && (
                <DashboardView events={events} sparkEPS={sparkEPS} sparkDetect={sparkDetect}
                  sparkScore={sparkScore} threshold={threshold} density={tweaks.density}
                  liveUpdates={tweaks.liveUpdates} onDismiss={dismissEv}
                  modelMetrics={modelMetrics} attackRecall={attackRecall} />
              )}
              {nav === 'model' && (
                <ModelView modelMetrics={modelMetrics} attackRecall={attackRecall}
                  tuningResults={tuningResults} ablation={data.ablation} config={data.metrics.config} />
              )}
              {nav === 'events' && (
                <EventsView events={events} threshold={threshold} density={tweaks.density} onDismiss={dismissEv} />
              )}
              {nav === 'network' && <NetworkView events={events} />}
              {nav === 'reports' && (
                <ReportsView events={events} modelMetrics={modelMetrics} attackRecall={attackRecall} threshold={threshold} />
              )}
            </>
          )}
        </div>
      </div>
      <TweaksPanel visible={tweaksOpen} tweaks={tweaks} onChange={setTweak}
        threshold={threshold} onThresholdChange={setUserThreshold} />
    </div>
  );
}
