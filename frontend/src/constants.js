export const ATTACK_CATS = ['DoS', 'DDoS', 'PortScan', 'Brute Force', 'Web Attack', 'Botnet'];

export const CAT_COLOR = {
  DoS: 'var(--red)',
  DDoS: 'oklch(0.67 0.22 14)',
  PortScan: 'var(--cyan)',
  'Brute Force': 'var(--amber)',
  'Web Attack': 'oklch(0.72 0.14 90)',
  Botnet: 'var(--violet)',
  BENIGN: 'var(--green)',
};

export const CAT_BG = {
  DoS: 'oklch(0.64 0.210 22/0.12)',
  DDoS: 'oklch(0.67 0.22 14/0.12)',
  PortScan: 'var(--cyan-bg)',
  'Brute Force': 'var(--amber-bg)',
  'Web Attack': 'oklch(0.72 0.14 90/0.10)',
  Botnet: 'var(--violet-bg)',
  BENIGN: 'var(--green-bg)',
};

export const DOS_SUB = ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'];
export const WEB_SUB = ['XSS Injection', 'SQL Injection', 'Brute Force (Web)'];
export const BF_SUB = ['FTP-Patator', 'SSH-Patator'];
export const SRC_IPS = [
  '185.220.101.47', '45.33.32.156', '194.165.16.11', '77.247.181.163',
  '91.108.4.0', '23.129.64.218', '162.55.18.200', '5.199.130.190',
  '10.0.1.55', '172.16.0.23',
];
export const GEOS = ['RU', 'CN', 'IR', 'KP', 'UA', 'DE', 'US', 'BR', 'NG', 'VN'];
export const HOSTS = [
  'ws-proc-01', 'db-master-2', 'api-gateway', 'auth-svc', 'k8s-node-07',
  'mail-relay', 'vpn-edge', 'lb-prod-01', 'storage-nas', 'hr-ws-14',
];

export const ri = (a) => a[Math.floor(Math.random() * a.length)];
export const rInt = (a, b) => Math.floor(Math.random() * (b - a + 1)) + a;
export const rF = (a, b) => Math.random() * (b - a) + a;
export const fmtK = (n) =>
  n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : String(n);
export const f4 = (n) => (n ?? 0).toFixed(4);
export const pct = (n) => ((n ?? 0) * 100).toFixed(2) + '%';
export const fmtTs = (d) =>
  d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
export const fmtFull = (d) =>
  `${d.toLocaleDateString('en-US', { month: 'short', day: '2-digit' })} ${fmtTs(d)}`;

export const catSub = (c) =>
  c === 'DoS' ? ri(DOS_SUB) : c === 'Web Attack' ? ri(WEB_SUB) : c === 'Brute Force' ? ri(BF_SUB) : c;

export const gauss = (mu, sigma) => {
  const u = 1 - Math.random(), v = Math.random();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

let _id = 9000;
// Generate a synthetic event. Anomaly scores are drawn from the
// per-category Gaussian (mean/std from val split) so the distribution
// mirrors what IForest actually produces — attacks with real signal
// (DoS/DDoS) land above the threshold, stealthy categories overlap BENIGN.
export const mkEv = (stats, threshold) => {
  const benign = Math.random() < 0.80;
  const cat = benign ? 'BENIGN' : ri(ATTACK_CATS);
  const s = stats[cat];
  const score = parseFloat(gauss(s.score_mean, s.score_std).toFixed(4));
  const flagged = score >= threshold;
  const status = benign ? (flagged ? 'FP' : 'OK') : (flagged ? 'DETECTED' : 'MISSED');
  return {
    id: ++_id, ts: new Date(), cat, sub: catSub(cat),
    src: ri(SRC_IPS), dst: ri(HOSTS), geo: ri(GEOS),
    score, flagged, status,
  };
};
