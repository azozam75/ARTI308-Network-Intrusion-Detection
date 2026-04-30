import { CAT_BG, CAT_COLOR } from './constants';

export const Sparkline = ({ data, color = 'var(--amber)', W = 80, H = 26 }) => {
  const max = Math.max(...data, 0.001), min = Math.min(...data), r = max - min || 0.001;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * W},${H - ((v - min) / r) * (H - 3) - 1}`).join(' ');
  return (
    <svg width={W} height={H} style={{ display: 'block', flexShrink: 0 }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" opacity="0.85" />
    </svg>
  );
};

export const CatBadge = ({ cat, size = 'sm' }) => {
  const fs = size === 'lg' ? '11px' : '10px';
  const pad = size === 'lg' ? '3px 9px' : '2px 7px';
  return (
    <span style={{
      font: `600 ${fs} var(--mono)`, letterSpacing: '0.07em', padding: pad, borderRadius: 3,
      color: CAT_COLOR[cat] || 'var(--muted)', background: CAT_BG[cat] || 'transparent',
      border: `1px solid ${(CAT_COLOR[cat] || 'var(--muted)')}55`, whiteSpace: 'nowrap',
      animation: (cat === 'DDoS' || cat === 'DoS') ? 'pulse-red 2.2s infinite' : 'none',
    }}>{cat}</span>
  );
};

export const SectionHead = ({ title, right }) => (
  <div style={{
    padding: '8px 14px', borderBottom: '1px solid var(--border)',
    background: 'var(--surface2)', display: 'flex', justifyContent: 'space-between',
    alignItems: 'center', flexShrink: 0,
  }}>
    <span style={{ font: '600 10px var(--sans)', letterSpacing: '0.13em', textTransform: 'uppercase', color: 'var(--muted)' }}>{title}</span>
    {right}
  </div>
);

export const MetricCard = ({ label, value, sub, color = 'var(--amber)', spark, alert }) => (
  <div style={{
    background: 'var(--surface)', border: `1px solid ${alert ? color : 'var(--border)'}`,
    padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 6,
    position: 'relative', overflow: 'hidden',
    boxShadow: alert ? `0 0 20px ${color}18` : 'none',
  }}>
    {alert && <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg,transparent,${color},transparent)` }} />}
    <span style={{ font: '500 10px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)' }}>{label}</span>
    <span style={{ font: '700 28px var(--mono)', color, lineHeight: 1 }}>{value}</span>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 8 }}>
      {sub && <span style={{ font: '400 11px var(--mono)', color: 'var(--dim)', lineHeight: 1.3 }}>{sub}</span>}
      {spark && <Sparkline data={spark} color={color} />}
    </div>
  </div>
);
