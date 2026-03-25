import json
import os
from flask import Flask, jsonify, render_template_string, send_from_directory

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Walker Training Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Courier New', monospace; background: #0d0d0d; color: #e0e0e0; padding: 24px; }
    h1 { color: #4aadff; font-size: 22px; margin-bottom: 4px; }
    .subtitle { color: #666; font-size: 12px; margin-bottom: 24px; }
    .status-bar {
      background: #1a1a1a; border-left: 3px solid #4aadff;
      padding: 10px 16px; margin-bottom: 24px; font-size: 13px; color: #aaa;
      border-radius: 0 4px 4px 0;
    }
    .status-bar span { color: #fff; font-weight: bold; }
    .charts { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 32px; }
    .chart-box {
      background: #1a1a1a; border-radius: 8px; padding: 20px;
      flex: 1; min-width: 380px; border: 1px solid #2a2a2a;
    }
    .chart-box h3 { color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
    canvas { max-height: 240px; }
    .snapshots h2 { color: #4aadff; font-size: 16px; margin-bottom: 16px; }
    .snap-grid { display: flex; flex-wrap: wrap; gap: 16px; }
    .snap-card {
      background: #1a1a1a; border-radius: 8px; padding: 14px;
      border: 1px solid #2a2a2a; min-width: 320px;
    }
    .snap-card .step-label { color: #4aadff; font-size: 12px; margin-bottom: 8px; font-weight: bold; }
    .snap-card .reward-label { color: #888; font-size: 11px; margin-top: 6px; }
    video { border-radius: 4px; display: block; width: 100%; }
    .empty { color: #444; font-style: italic; padding: 20px 0; }
    .refresh-hint { color: #333; font-size: 11px; margin-top: 24px; text-align: right; }
  </style>
</head>
<body>
  <h1>Walker Training Dashboard</h1>
  <div class="subtitle">BipedalWalker-v3 · PPO · auto-refreshes every 10s</div>

  <div class="status-bar" id="statusBar">Waiting for data...</div>

  <div class="charts">
    <div class="chart-box">
      <h3>Mean Episode Reward</h3>
      <canvas id="rewardChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Mean Episode Length</h3>
      <canvas id="lenChart"></canvas>
    </div>
  </div>

  <div class="snapshots">
    <h2>Snapshots</h2>
    <div class="snap-grid" id="snapGrid"><div class="empty">No snapshots yet.</div></div>
  </div>

  <div class="refresh-hint" id="refreshHint"></div>

<script>
const mkChart = (id, label, color) => new Chart(document.getElementById(id), {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label,
      data: [],
      borderColor: color,
      backgroundColor: color + '18',
      borderWidth: 2,
      pointRadius: 3,
      fill: true,
      tension: 0.3,
    }]
  },
  options: {
    animation: false,
    responsive: true,
    scales: {
      x: { ticks: { color: '#666', font: { size: 10 } }, grid: { color: '#1f1f1f' } },
      y: { ticks: { color: '#666', font: { size: 10 } }, grid: { color: '#1f1f1f' } },
    },
    plugins: { legend: { display: false } },
  }
});

const rewardChart = mkChart('rewardChart', 'Mean Reward', '#4aadff');
const lenChart    = mkChart('lenChart',    'Mean Ep Len', '#ffaa44');

// Track which snapshots have already been rendered to avoid DOM churn
const renderedSnaps = new Set();

async function refresh() {
  try {
    const [metrics, snaps] = await Promise.all([
      fetch('/api/metrics').then(r => r.json()),
      fetch('/api/snapshots').then(r => r.json()),
    ]);

    // --- Charts ---
    rewardChart.data.labels                  = metrics.map(m => m.step.toLocaleString());
    rewardChart.data.datasets[0].data        = metrics.map(m => +m.mean_reward.toFixed(1));
    rewardChart.update();
    lenChart.data.labels                     = metrics.map(m => m.step.toLocaleString());
    lenChart.data.datasets[0].data           = metrics.map(m => +m.mean_ep_len.toFixed(0));
    lenChart.update();

    // --- Status bar ---
    const last = metrics[metrics.length - 1];
    if (last) {
      document.getElementById('statusBar').innerHTML =
        `Step <span>${last.step.toLocaleString()}</span> &nbsp;|&nbsp; ` +
        `Reward <span>${last.mean_reward.toFixed(1)}</span> &nbsp;|&nbsp; ` +
        `Ep len <span>${last.mean_ep_len.toFixed(0)}</span> &nbsp;|&nbsp; ` +
        `Checkpoints <span>${snaps.length}</span>`;
    }

    // --- Snapshots (newest first, only add new ones) ---
    const grid = document.getElementById('snapGrid');
    const sorted = [...snaps].sort((a, b) => b.step - a.step);

    if (sorted.length === 0) return;
    if (grid.querySelector('.empty')) grid.innerHTML = '';

    sorted.forEach(snap => {
      if (renderedSnaps.has(snap.filename)) return;
      renderedSnaps.add(snap.filename);

      const metricEntry = metrics.find(m => m.step === snap.step);
      const rewardLabel = metricEntry ? `reward: ${metricEntry.mean_reward.toFixed(1)}` : '';

      const card = document.createElement('div');
      card.className = 'snap-card';
      card.innerHTML = `
        <div class="step-label">Step ${snap.step.toLocaleString()}</div>
        <video controls loop muted playsinline>
          <source src="/snapshots/${snap.filename}" type="video/mp4">
        </video>
        <div class="reward-label">${rewardLabel}</div>`;
      grid.prepend(card);
    });

    document.getElementById('refreshHint').textContent =
      `Last refresh: ${new Date().toLocaleTimeString()}`;

  } catch (e) {
    console.warn('Refresh failed:', e);
  }
}

refresh();
setInterval(refresh, 10_000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/metrics")
def metrics():
    if not os.path.exists("metrics.json"):
        return jsonify([])
    with open("metrics.json") as f:
        return jsonify(json.load(f))


@app.route("/api/snapshots")
def snapshots():
    snap_dir = "snapshots"
    if not os.path.exists(snap_dir):
        return jsonify([])
    files = sorted(f for f in os.listdir(snap_dir) if f.endswith(".mp4"))
    result = []
    for fname in files:
        # filename: step_00025000.mp4
        step_str = fname.replace("step_", "").replace(".mp4", "").lstrip("0") or "0"
        result.append({"step": int(step_str), "filename": fname})
    return jsonify(result)


@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(os.path.abspath("snapshots"), filename)


if __name__ == "__main__":
    print("Dashboard running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
