import { useMemo, useState } from 'react';

import { ATTACK_CATS, CAT_BG, CAT_COLOR, f4, fmtFull, fmtK, fmtTs, pct } from './constants';
import { CatBadge, MetricCard, SectionHead } from './components';
import { imageUrl } from './api';

/* ─── PANELS (used by the Dashboard overview) ───────────── */

export const AttackRecallPanel = ({ attackRecall }) => (
  <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
    <SectionHead title="Per-Attack Recall" right={<span style={{ font: '400 10px var(--mono)', color: 'var(--dim)' }}>CIC-IDS-2017 · val split</span>} />
    <div style={{ flex: 1, padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, overflowY: 'auto' }}>
      {ATTACK_CATS.map((cat) => {
        const d = attackRecall[cat], r = d?.recall ?? 0, c = CAT_COLOR[cat];
        return (
          <div key={cat}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <CatBadge cat={cat} />
                <span style={{ font: '400 10px var(--mono)', color: 'var(--dim)' }}>n={fmtK(d?.n || 0)}</span>
              </div>
              <span style={{ font: '700 13px var(--mono)', color: r >= 0.95 ? 'var(--green)' : r >= 0.85 ? 'var(--amber)' : 'var(--red)' }}>{pct(r)}</span>
            </div>
            <div style={{ height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${r * 100}%`, background: c, borderRadius: 3, opacity: 0.75 }} />
            </div>
          </div>
        );
      })}
      <div style={{ paddingTop: 8, borderTop: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <span style={{ font: '500 10px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--muted)' }}>BENIGN — FPR</span>
          <span style={{ font: '700 13px var(--mono)', color: 'var(--green)' }}>{pct(attackRecall.BENIGN?.false_positive_rate ?? 0)}</span>
        </div>
        <div style={{ height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${(attackRecall.BENIGN?.false_positive_rate ?? 0) * 100}%`, background: 'var(--red)', opacity: 0.7, borderRadius: 3 }} />
        </div>
      </div>
    </div>
  </div>
);

const SCORE_MIN = -0.25, SCORE_MAX = 0.20;
export const ScoreHistogram = ({ events, threshold }) => {
  const bins = 40, bArr = new Array(bins).fill(0), aArr = new Array(bins).fill(0);
  const span = SCORE_MAX - SCORE_MIN;
  events.forEach((e) => {
    const norm = (e.score - SCORE_MIN) / span;
    const i = Math.max(0, Math.min(Math.floor(norm * bins), bins - 1));
    (e.cat === 'BENIGN' ? bArr : aArr)[i]++;
  });
  const max = Math.max(...bArr, ...aArr, 1), W = 100, H = 50;
  const bW = W / bins, tX = ((threshold - SCORE_MIN) / span) * W;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <SectionHead title="Anomaly Score Distribution" right={<span style={{ font: '500 11px var(--mono)', color: 'var(--amber)' }}>thr = {threshold.toFixed(4)}</span>} />
      <div style={{ flex: 1, padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <svg viewBox={`0 0 100 ${H}`} preserveAspectRatio="none" style={{ width: '100%', height: H, display: 'block' }}>
          {bArr.map((v, i) => <rect key={`b${i}`} x={i * bW} y={H - (v / max) * H} width={bW * 0.75} height={(v / max) * H} fill="var(--green)" opacity="0.4" />)}
          {aArr.map((v, i) => <rect key={`a${i}`} x={i * bW} y={H - (v / max) * H} width={bW * 0.75} height={(v / max) * H} fill="var(--amber)" opacity="0.65" />)}
          <line x1={tX} y1={0} x2={tX} y2={H} stroke="var(--red)" strokeWidth="0.6" strokeDasharray="2,1.5" />
        </svg>
        <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
          {[{ c: 'var(--green)', l: 'BENIGN' }, { c: 'var(--amber)', l: 'ATTACK' }, { c: 'var(--red)', l: 'THRESHOLD', line: true }].map((x) => (
            <span key={x.l} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              {x.line
                ? <span style={{ display: 'inline-block', width: 1, height: 12, background: x.c }} />
                : <span style={{ display: 'inline-block', width: 10, height: 10, background: x.c, borderRadius: 2, opacity: 0.7 }} />}
              <span style={{ font: '400 10px var(--mono)', color: 'var(--muted)' }}>{x.l}</span>
            </span>
          ))}
        </div>
        <div style={{ font: '400 10px var(--mono)', color: 'var(--dim)' }}>
          BENIGN ≈ -0.19..-0.12; DoS/DDoS shift right of threshold; stealthy attacks overlap BENIGN
        </div>
      </div>
    </div>
  );
};

export const ConfusionMini = ({ modelMetrics, threshold }) => {
  const { tn, fp, fn, tp } = modelMetrics.confusion_matrix;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <SectionHead title="Confusion Matrix" />
      <div style={{ flex: 1, padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          {[
            { k: 'True Positive', v: tp, c: 'var(--green)' },
            { k: 'False Positive', v: fp, c: 'var(--red)' },
            { k: 'False Negative', v: fn, c: 'var(--amber)' },
            { k: 'True Negative', v: tn, c: 'var(--cyan)' },
          ].map(({ k, v, c }) => (
            <div key={k} style={{ background: 'var(--surface3)', padding: '10px 12px', borderRadius: 3, border: `1px solid ${c}22` }}>
              <div style={{ font: '500 9px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 4 }}>{k}</div>
              <div style={{ font: '700 18px var(--mono)', color: c }}>{fmtK(v)}</div>
            </div>
          ))}
        </div>
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          <div style={{ font: '500 9px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 4 }}>Decision Threshold</div>
          <div style={{ font: '700 16px var(--mono)', color: 'var(--amber)' }}>{threshold.toFixed(6)}</div>
          <div style={{ font: '400 10px var(--mono)', color: 'var(--dim)', marginTop: 2 }}>grid-search · max F1 on val split</div>
        </div>
      </div>
    </div>
  );
};

/* ─── EVENTS VIEW ─────────────────────────────────────── */

export const EventsView = ({ events, threshold, density, onDismiss }) => {
  const [catF, setCatF] = useState(null);
  const [statF, setStatF] = useState(null);
  const [search, setSearch] = useState('');
  const [sortK, setSortK] = useState('ts');
  const [sortD, setSortD] = useState(-1);
  const rowH = density === 'compact' ? 30 : density === 'spacious' ? 46 : 38;
  const tS = (k) => { if (sortK === k) setSortD((d) => -d); else { setSortK(k); setSortD(-1); } };
  const rows = useMemo(
    () => [...events]
      .filter((e) => !catF || e.cat === catF)
      .filter((e) => !statF || e.status === statF)
      .filter((e) => !search || [e.sub, e.src, e.dst, e.cat].some((v) => v.toLowerCase().includes(search.toLowerCase())))
      .sort((a, b) => { const av = a[sortK], bv = b[sortK]; return av < bv ? -sortD : av > bv ? sortD : 0; }),
    [events, catF, statF, search, sortK, sortD]
  );

  const statColor = { DETECTED: 'var(--green)', MISSED: 'var(--red)', FP: 'var(--amber)', OK: 'var(--cyan)' };
  const headCell = (k, label, w) => (
    <th key={k} onClick={() => tS(k)} style={{
      font: '600 10px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase',
      color: sortK === k ? 'var(--amber)' : 'var(--muted)', cursor: 'pointer',
      padding: '0 10px', textAlign: 'left', whiteSpace: 'nowrap', userSelect: 'none', width: w,
    }}>{label}{sortK === k ? (sortD > 0 ? ' ↑' : ' ↓') : ''}</th>
  );

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{
        padding: '10px 14px', borderBottom: '1px solid var(--border)', background: 'var(--surface2)',
        display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0, flexWrap: 'wrap',
      }}>
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search events…"
          style={{
            background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)',
            font: '400 12px var(--mono)', padding: '5px 10px', borderRadius: 3, width: 180, outline: 'none',
          }} />
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {ATTACK_CATS.concat(['BENIGN']).map((c) => (
            <button key={c} onClick={() => setCatF((f) => f === c ? null : c)} style={{
              font: '600 9px var(--mono)', letterSpacing: '0.06em', padding: '3px 8px', borderRadius: 3, cursor: 'pointer',
              background: catF === c ? CAT_BG[c] : 'transparent',
              border: `1px solid ${catF === c ? (CAT_COLOR[c] + '88') : 'var(--border)'}`,
              color: catF === c ? CAT_COLOR[c] : 'var(--dim)',
            }}>{c}</button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          {['DETECTED', 'MISSED', 'FP', 'OK'].map((s) => (
            <button key={s} onClick={() => setStatF((f) => f === s ? null : s)} style={{
              font: '600 9px var(--mono)', letterSpacing: '0.06em', padding: '3px 8px', borderRadius: 3, cursor: 'pointer',
              background: statF === s ? `${statColor[s]}18` : 'transparent',
              border: `1px solid ${statF === s ? (statColor[s] + '88') : 'var(--border)'}`,
              color: statF === s ? statColor[s] : 'var(--dim)',
            }}>{s}</button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <span style={{ font: '400 11px var(--mono)', color: 'var(--muted)' }}>{rows.length} / {events.length} events</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
          <colgroup>
            <col style={{ width: 5 }} /><col style={{ width: 92 }} /><col style={{ width: 88 }} /><col />
            <col style={{ width: 130 }} /><col style={{ width: 110 }} /><col style={{ width: 36 }} />
            <col style={{ width: 80 }} /><col style={{ width: 90 }} /><col style={{ width: 24 }} />
          </colgroup>
          <thead>
            <tr style={{ background: 'var(--surface)', height: 32, position: 'sticky', top: 0, zIndex: 2 }}>
              <th />
              {headCell('ts', 'Timestamp')}
              {headCell('cat', 'Category')}
              {headCell('sub', 'Subtype')}
              {headCell('src', 'Source IP')}
              {headCell('dst', 'Endpoint')}
              <th style={{ font: '600 10px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--muted)', padding: '0 10px', textAlign: 'left' }}>GEO</th>
              {headCell('score', 'Score')}
              {headCell('status', 'Status')}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((ev, i) => {
              const sc = statColor[ev.status], isAnom = ev.score >= threshold;
              return (
                <tr key={ev.id} style={{
                  height: rowH,
                  background: i % 2 === 0 ? 'transparent' : 'oklch(0.14 0.018 245/0.4)',
                  cursor: 'default',
                  borderLeft: ev.status === 'DETECTED' ? '3px solid var(--green)33'
                    : ev.status === 'MISSED' ? '3px solid var(--red)33' : '3px solid transparent',
                }}>
                  <td />
                  <td style={{ font: '400 11px var(--mono)', color: 'var(--muted)', padding: '0 10px', whiteSpace: 'nowrap' }}>{fmtTs(ev.ts)}</td>
                  <td style={{ padding: '0 10px' }}>
                    {ev.cat !== 'BENIGN' ? <CatBadge cat={ev.cat} /> : <span style={{ font: '400 11px var(--mono)', color: 'var(--dim)' }}>BENIGN</span>}
                  </td>
                  <td style={{ font: '500 12px var(--sans)', color: ev.cat === 'BENIGN' ? 'var(--dim)' : 'var(--text)', padding: '0 10px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ev.sub}</td>
                  <td style={{ font: '400 11px var(--mono)', color: 'var(--cyan)', padding: '0 10px', whiteSpace: 'nowrap' }}>{ev.src}</td>
                  <td style={{ font: '400 11px var(--mono)', color: 'var(--muted)', padding: '0 10px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ev.dst}</td>
                  <td style={{ font: '700 10px var(--mono)', color: 'var(--dim)', padding: '0 10px', letterSpacing: '0.05em' }}>{ev.geo}</td>
                  <td style={{ padding: '0 10px' }}>
                    <span style={{ font: '600 12px var(--mono)', color: isAnom ? (ev.score > 0.05 ? 'var(--red)' : 'var(--amber)') : 'var(--dim)' }}>{ev.score.toFixed(4)}</span>
                  </td>
                  <td style={{ padding: '0 10px' }}>
                    <span style={{ font: '600 11px var(--mono)', color: sc, letterSpacing: '0.06em' }}>{ev.status}</span>
                  </td>
                  <td style={{ padding: '0 4px' }}>
                    <button onClick={() => onDismiss(ev.id)} style={{ background: 'transparent', border: 'none', color: 'var(--dim)', cursor: 'pointer', font: '11px var(--mono)', padding: 2 }}>✕</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/* ─── DASHBOARD VIEW (overview) ───────────────────────── */

export const DashboardView = ({ events, sparkEPS, sparkDetect, sparkScore, threshold, density, liveUpdates, onDismiss, modelMetrics, attackRecall }) => {
  const detectedCount = events.filter((e) => e.status === 'DETECTED').length;
  const fpCount = events.filter((e) => e.status === 'FP').length;
  const missedCount = events.filter((e) => e.status === 'MISSED').length;
  const liveF1 = detectedCount / (detectedCount + fpCount / 2 + missedCount + 0.001);
  const avgScore = events.length > 0 ? (events.reduce((s, e) => s + e.score, 0) / events.length).toFixed(4) : 0;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 1, background: 'var(--border)', flexShrink: 0 }}>
        <MetricCard label="Events / sec" value={fmtK(sparkEPS[sparkEPS.length - 1])} sub="flow classification" color="var(--cyan)" spark={sparkEPS} />
        <MetricCard label="Live F1" value={liveF1.toFixed(4)} sub="rolling session window" color="var(--amber)" spark={sparkDetect} alert={liveF1 < 0.88} />
        <MetricCard label="ROC-AUC" value={f4(modelMetrics.roc_auc)} sub="val set · IForest" color="var(--violet)" />
        <MetricCard label="Avg Anomaly Score" value={avgScore} sub={`thr = ${threshold.toFixed(4)}`} color={avgScore > threshold ? 'var(--red)' : 'var(--amber)'} spark={sparkScore} />
        <MetricCard label="FPR" value={pct(modelMetrics.false_positive_rate)} sub="benign mis-flagged" color="var(--green)" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 240px', gap: 1, background: 'var(--border)', height: 220, flexShrink: 0 }}>
        <div style={{ background: 'var(--surface)', overflow: 'hidden' }}><AttackRecallPanel attackRecall={attackRecall} /></div>
        <div style={{ background: 'var(--surface)', overflow: 'hidden' }}><ScoreHistogram events={events} threshold={threshold} /></div>
        <div style={{ background: 'var(--surface)', overflow: 'hidden' }}><ConfusionMini modelMetrics={modelMetrics} threshold={threshold} /></div>
      </div>

      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 240px', gap: 1, background: 'var(--border)', overflow: 'hidden' }}>
        <div style={{ background: 'var(--surface)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <SectionHead title="Live Event Feed — IForest Predictions"
            right={liveUpdates && (
              <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--green)' }}>
                <span style={{ animation: 'blink 1s steps(1) infinite', fontSize: 9 }}>●</span>
                <span style={{ font: '500 10px var(--mono)' }}>LIVE</span>
              </span>
            )} />
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <EventsView events={events.slice(0, 60)} threshold={threshold} density={density} onDismiss={onDismiss} />
          </div>
        </div>
        <div style={{ background: 'var(--surface)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <SectionHead title="Session Stats" />
          <div style={{ flex: 1, padding: '14px', display: 'flex', flexDirection: 'column', gap: 10, overflowY: 'auto' }}>
            {[
              { l: 'Total Events', v: events.length, c: 'var(--text)' },
              { l: 'Attacks', v: events.filter((e) => e.cat !== 'BENIGN').length, c: 'var(--amber)' },
              { l: 'Detected', v: detectedCount, c: 'var(--green)' },
              { l: 'Missed', v: missedCount, c: 'var(--red)' },
              { l: 'False Positives', v: fpCount, c: 'var(--amber)' },
              { l: 'Benign OK', v: events.filter((e) => e.status === 'OK').length, c: 'var(--cyan)' },
            ].map(({ l, v, c }) => (
              <div key={l} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
                <span style={{ font: '500 12px var(--sans)', color: 'var(--muted)' }}>{l}</span>
                <span style={{ font: '700 16px var(--mono)', color: c }}>{v}</span>
              </div>
            ))}
            <div style={{ marginTop: 4 }}>
              <div style={{ font: '500 10px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 6 }}>Attack Breakdown</div>
              {ATTACK_CATS.map((cat) => {
                const n = events.filter((e) => e.cat === cat).length;
                if (!n) return null;
                return (
                  <div key={cat} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <CatBadge cat={cat} />
                    <span style={{ font: '600 13px var(--mono)', color: CAT_COLOR[cat] }}>{n}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─── MODEL VIEW ──────────────────────────────────────── */

export const ModelView = ({ modelMetrics, attackRecall, tuningResults, ablation, config }) => {
  const rows = [
    { k: 'Accuracy', v: modelMetrics.accuracy, c: 'var(--cyan)' },
    { k: 'Precision', v: modelMetrics.precision, c: 'var(--green)' },
    { k: 'Recall', v: modelMetrics.recall, c: 'var(--amber)' },
    { k: 'F1 Score', v: modelMetrics.f1, c: 'var(--amber)' },
    { k: 'F1 Macro', v: modelMetrics.f1_macro, c: 'var(--amber2)' },
    { k: 'ROC-AUC', v: modelMetrics.roc_auc, c: 'var(--violet)' },
    { k: 'FPR', v: modelMetrics.false_positive_rate, c: 'var(--red)', invert: true },
    { k: 'TPR', v: modelMetrics.true_positive_rate, c: 'var(--green)' },
  ];
  const configRows = [
    ['Model', 'Isolation Forest'],
    ['n_estimators', String(config.n_estimators)],
    ['max_samples', String(config.max_samples)],
    ['contamination', String(config.contamination)],
    ['max_features', String(config.max_features)],
    ['bootstrap', String(config.bootstrap)],
    ['random_state', String(config.random_state)],
    ['n_features selected', '50'],
    ['train_subsample', String(config.train_subsample)],
    ['threshold grid', '401 pts'],
    ['split (train/val/test)', '70 / 15 / 15'],
    ['dataset', 'CIC-IDS-2017'],
  ];

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', background: 'var(--surface2)', flexShrink: 0 }}>
        <div style={{ font: '700 16px var(--sans)', color: 'var(--text)' }}>Model Performance Report</div>
        <div style={{ font: '400 12px var(--mono)', color: 'var(--muted)', marginTop: 3 }}>Isolation Forest · Semi-supervised · Trained on BENIGN-only rows</div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>

        <div>
          <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>Evaluation Metrics — Validation Set</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {rows.map((r) => (
              <div key={r.k} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 72px', alignItems: 'center', gap: 12 }}>
                <span style={{ font: '500 12px var(--sans)', color: 'var(--text)' }}>{r.k}</span>
                <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(r.invert ? (1 - r.v) : r.v) * 100}%`, background: r.c, borderRadius: 4, opacity: 0.75, transition: 'width 0.9s' }} />
                </div>
                <span style={{ font: '700 13px var(--mono)', color: r.c, textAlign: 'right' }}>{f4(r.v)}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>Per-Attack Detection Recall</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {ATTACK_CATS.map((cat) => {
              const d = attackRecall[cat], r = d?.recall ?? 0;
              return (
                <div key={cat} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 72px 80px', alignItems: 'center', gap: 12 }}>
                  <CatBadge cat={cat} size="lg" />
                  <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${r * 100}%`, background: CAT_COLOR[cat], borderRadius: 4, opacity: 0.75 }} />
                  </div>
                  <span style={{ font: '700 13px var(--mono)', color: r >= 0.95 ? 'var(--green)' : r >= 0.85 ? 'var(--amber)' : 'var(--red)' }}>{pct(r)}</span>
                  <span style={{ font: '400 11px var(--mono)', color: 'var(--dim)' }}>n={fmtK(d?.n || 0)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {ablation && (
          <div>
            <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>
              Week-9 Ablation — Per-Category Thresholding
              <span style={{ font: '400 10px var(--mono)', color: 'var(--dim)', marginLeft: 10, textTransform: 'none', letterSpacing: 'normal' }}>
                oracle upper bound · macro recall {pct(ablation.macro_per_category.recall)} vs global {pct(ablation.macro_global.recall)}
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {ATTACK_CATS.map((cat) => {
                const row = ablation.categories[cat];
                if (!row) return null;
                const gR = row.global.recall, pR = row.per_category.recall;
                return (
                  <div key={cat} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr 100px', alignItems: 'center', gap: 12 }}>
                    <CatBadge cat={cat} size="lg" />
                    <div>
                      <div style={{ font: '400 10px var(--mono)', color: 'var(--dim)' }}>global</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ flex: 1, height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${gR * 100}%`, background: 'var(--dim)', borderRadius: 3 }} />
                        </div>
                        <span style={{ font: '600 11px var(--mono)', color: 'var(--muted)', minWidth: 48, textAlign: 'right' }}>{pct(gR)}</span>
                      </div>
                    </div>
                    <div>
                      <div style={{ font: '400 10px var(--mono)', color: 'var(--dim)' }}>per-category</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ flex: 1, height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${pR * 100}%`, background: CAT_COLOR[cat], borderRadius: 3, opacity: 0.75 }} />
                        </div>
                        <span style={{ font: '600 11px var(--mono)', color: CAT_COLOR[cat], minWidth: 48, textAlign: 'right' }}>{pct(pR)}</span>
                      </div>
                    </div>
                    <span style={{ font: '400 10px var(--mono)', color: 'var(--dim)', textAlign: 'right' }}>AUC {f4(row.test_auc)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div>
          <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>Model Config — ModelConfig</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 6 }}>
            {configRows.map(([k, v]) => (
              <div key={k} style={{ background: 'var(--surface2)', border: '1px solid var(--border)', padding: '8px 10px', borderRadius: 3 }}>
                <div style={{ font: '500 9px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--dim)', marginBottom: 3 }}>{k}</div>
                <div style={{ font: '600 13px var(--mono)', color: 'var(--text)' }}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>Confusion Matrix — Validation Set</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
            {[
              { k: 'True Positive', v: modelMetrics.confusion_matrix.tp, c: 'var(--green)' },
              { k: 'False Positive', v: modelMetrics.confusion_matrix.fp, c: 'var(--red)' },
              { k: 'False Negative', v: modelMetrics.confusion_matrix.fn, c: 'var(--amber)' },
              { k: 'True Negative', v: modelMetrics.confusion_matrix.tn, c: 'var(--cyan)' },
            ].map(({ k, v, c }) => (
              <div key={k} style={{ background: 'var(--surface2)', border: `1px solid ${c}33`, padding: '12px 14px', borderRadius: 3 }}>
                <div style={{ font: '500 10px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 6 }}>{k}</div>
                <div style={{ font: '700 22px var(--mono)', color: c }}>{fmtK(v)}</div>
              </div>
            ))}
          </div>
        </div>

        {tuningResults && tuningResults.results && (
          <div>
            <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>
              Tuning Leaderboard — top 5 of {tuningResults.results.length}
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', font: '400 12px var(--mono)' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Rank', 'n_est', 'max_samples', 'max_feat', 'F1', 'ROC-AUC', 'FPR'].map((h) => (
                    <th key={h} style={{ font: '600 10px var(--sans)', letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--muted)', padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...tuningResults.results].sort((a, b) => b.val_metrics.f1 - a.val_metrics.f1).slice(0, 5).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)', background: i === 0 ? 'var(--amber-bg)' : 'transparent' }}>
                    <td style={{ padding: '7px 10px', color: i === 0 ? 'var(--amber)' : 'var(--muted)' }}>{i + 1}</td>
                    <td style={{ padding: '7px 10px' }}>{r.config.n_estimators}</td>
                    <td style={{ padding: '7px 10px' }}>{r.config.max_samples}</td>
                    <td style={{ padding: '7px 10px' }}>{r.config.max_features}</td>
                    <td style={{ padding: '7px 10px', color: 'var(--amber)' }}>{f4(r.val_metrics.f1)}</td>
                    <td style={{ padding: '7px 10px' }}>{f4(r.val_metrics.roc_auc)}</td>
                    <td style={{ padding: '7px 10px' }}>{pct(r.val_metrics.false_positive_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </div>
  );
};

/* ─── NETWORK VIEW ────────────────────────────────────── */

export const NetworkView = ({ events }) => {
  const srcCounts = {}, dstCounts = {}, catCounts = {};
  events.filter((e) => e.cat !== 'BENIGN').forEach((e) => {
    srcCounts[e.src] = (srcCounts[e.src] || 0) + 1;
    dstCounts[e.dst] = (dstCounts[e.dst] || 0) + 1;
    catCounts[e.cat] = (catCounts[e.cat] || 0) + 1;
  });
  const topSrc = Object.entries(srcCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const topDst = Object.entries(dstCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const maxSrc = topSrc[0]?.[1] || 1, maxDst = topDst[0]?.[1] || 1;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', background: 'var(--surface2)', flexShrink: 0 }}>
        <div style={{ font: '700 16px var(--sans)', color: 'var(--text)' }}>Network Analysis</div>
        <div style={{ font: '400 12px var(--mono)', color: 'var(--muted)', marginTop: 3 }}>Attack traffic source / destination breakdown from live event stream</div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div style={{ gridColumn: '1/-1' }}>
          <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 12 }}>Attack Category Distribution</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {Object.entries(catCounts).sort((a, b) => b[1] - a[1]).map(([cat, n]) => {
              const total = Object.values(catCounts).reduce((s, v) => s + v, 0);
              return (
                <div key={cat} style={{ background: 'var(--surface2)', border: `1px solid ${CAT_COLOR[cat]}44`, padding: '10px 16px', borderRadius: 4, minWidth: 120 }}>
                  <CatBadge cat={cat} size="lg" />
                  <div style={{ font: '700 22px var(--mono)', color: CAT_COLOR[cat], marginTop: 6 }}>{n}</div>
                  <div style={{ font: '400 10px var(--mono)', color: 'var(--dim)', marginTop: 2 }}>{((n / total) * 100).toFixed(1)}% of attacks</div>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>Top Source IPs</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {topSrc.map(([ip, n], i) => (
              <div key={ip}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ font: '700 10px var(--mono)', color: 'var(--dim)', minWidth: 18 }}>{String(i + 1).padStart(2, '0')}</span>
                    <span style={{ font: '500 12px var(--mono)', color: 'var(--cyan)' }}>{ip}</span>
                  </div>
                  <span style={{ font: '700 13px var(--mono)', color: 'var(--amber)' }}>{n}</span>
                </div>
                <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(n / maxSrc) * 100}%`, background: i === 0 ? 'var(--red)' : 'var(--amber)', borderRadius: 2, opacity: 0.75 }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 10 }}>Most Targeted Endpoints</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {topDst.map(([host, n], i) => (
              <div key={host}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ font: '700 10px var(--mono)', color: 'var(--dim)', minWidth: 18 }}>{String(i + 1).padStart(2, '0')}</span>
                    <span style={{ font: '500 12px var(--mono)', color: 'var(--text)' }}>{host}</span>
                  </div>
                  <span style={{ font: '700 13px var(--mono)', color: 'var(--violet)' }}>{n}</span>
                </div>
                <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(n / maxDst) * 100}%`, background: 'var(--violet)', borderRadius: 2, opacity: 0.75 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─── REPORTS VIEW ────────────────────────────────────── */

export const ReportsView = ({ events, modelMetrics, attackRecall, threshold }) => {
  const attacks = events.filter((e) => e.cat !== 'BENIGN').length;
  const det = events.filter((e) => e.status === 'DETECTED').length;
  const missed = events.filter((e) => e.status === 'MISSED').length;
  const fp = events.filter((e) => e.status === 'FP').length;
  const sections = [
    {
      title: 'Executive Summary',
      content: [
        `SENTINEL detected ${det} of ${attacks} attack events in the current session window.`,
        `${missed} attacks were missed (below decision threshold of ${threshold.toFixed(4)}).`,
        `${fp} benign flows were incorrectly flagged as anomalous (false positives).`,
        `Model precision on this window: ${attacks > 0 ? ((det / (det + fp + 0.001)) * 100).toFixed(1) : 'N/A'}%.`,
      ],
    },
    {
      title: 'IForest Model — Key Metrics',
      rows: [
        ['Accuracy', f4(modelMetrics.accuracy)],
        ['Precision', f4(modelMetrics.precision)],
        ['Recall', f4(modelMetrics.recall)],
        ['F1 Score', f4(modelMetrics.f1)],
        ['ROC-AUC', f4(modelMetrics.roc_auc)],
        ['False Positive Rate', pct(modelMetrics.false_positive_rate)],
      ],
    },
    {
      title: 'Attack Coverage',
      rows: ATTACK_CATS.map((c) => [c, pct(attackRecall[c]?.recall ?? 0), `n=${fmtK(attackRecall[c]?.n || 0)}`]),
    },
    {
      title: 'Pipeline Configuration',
      rows: [
        ['Algorithm', 'Isolation Forest (sklearn)'],
        ['n_estimators', '200'],
        ['Features', '50 (post-correlation pruning)'],
        ['Training set', 'BENIGN-only (semi-supervised)'],
        ['Train subsample', '400,000 rows'],
        ['Threshold', 'Grid-search on val · max F1'],
        ['Dataset', 'CIC-IDS-2017 (Canadian Institute for Cybersecurity)'],
        ['Attack categories', 'DoS · DDoS · PortScan · Brute Force · Web Attack · Botnet'],
      ],
    },
  ];

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', background: 'var(--surface2)', flexShrink: 0 }}>
        <div style={{ font: '700 16px var(--sans)', color: 'var(--text)' }}>Session Report</div>
        <div style={{ font: '400 12px var(--mono)', color: 'var(--muted)', marginTop: 3 }}>Generated {fmtFull(new Date())} · ARTI 308 · Group 2 · IAU</div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: 20 }}>
        {sections.map((s) => (
          <div key={s.title} style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)', background: 'var(--surface3)' }}>
              <span style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)' }}>{s.title}</span>
            </div>
            <div style={{ padding: '14px 16px' }}>
              {s.content?.map((line, i) => (
                <div key={i} style={{ font: '400 13px var(--sans)', color: 'var(--text)', marginBottom: 6, lineHeight: 1.6, display: 'flex', gap: 8 }}>
                  <span style={{ color: 'var(--amber)', flexShrink: 0 }}>—</span>{line}
                </div>
              ))}
              {s.rows && (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <tbody>
                    {s.rows.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                        {row.map((cell, j) => (
                          <td key={j} style={{
                            padding: '7px 8px',
                            font: j === 0 ? '500 12px var(--sans)' : '400 12px var(--mono)',
                            color: j === 0 ? 'var(--muted)' : 'var(--text)',
                          }}>
                            {j === 1 && s.title.includes('Coverage')
                              ? <span style={{ color: parseFloat(cell) >= 95 ? 'var(--green)' : parseFloat(cell) >= 85 ? 'var(--amber)' : 'var(--red)' }}>{cell}</span>
                              : cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        ))}

        <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)', background: 'var(--surface3)' }}>
            <span style={{ font: '600 11px var(--sans)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--muted)' }}>Evaluation Figures</span>
          </div>
          <div style={{ padding: '14px 16px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px,1fr))', gap: 12 }}>
            {['roc_curve.png', 'confusion_matrix.png', 'score_distributions.png', 'per_category_ablation.png', 'class_balance.png', 'pca_scatter.png'].map((name) => (
              <a key={name} href={imageUrl(name)} target="_blank" rel="noreferrer"
                 style={{ display: 'block', background: 'var(--surface3)', border: '1px solid var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                <img src={imageUrl(name)} alt={name} style={{ width: '100%', display: 'block', background: '#fff' }} />
                <div style={{ padding: '6px 10px', font: '500 10px var(--mono)', color: 'var(--dim)' }}>{name}</div>
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
