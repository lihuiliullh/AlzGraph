// AlzGraph interactive KG explorer — vanilla canvas force-directed graph.
(() => {
  const LAYERS = ["gene", "biomarker", "stage", "treatment", "outcome"];
  const COLOR = {
    gene: "#7c3aed", biomarker: "#0ea5e9", stage: "#eab308",
    treatment: "#14b8a6", outcome: "#ef4444", unknown: "#64748b",
  };
  const canvas = document.getElementById("kg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const info = document.getElementById("kg-info");
  const search = document.getElementById("kg-search");
  const legend = document.getElementById("kg-legend");

  let nodes = [], links = [], byId = new Map();
  const hidden = new Set();
  let hover = null, selected = null, dragging = null;
  let transform = { s: 1, x: 0, y: 0 };

  fetch("./data/demo_graph.json").then((r) => r.json()).then((g) => {
    document.querySelectorAll('[data-stat="entities"]').forEach((e) => (e.textContent = g.meta.nodes));
    document.querySelectorAll('[data-stat="triplets"]').forEach((e) => (e.textContent = g.meta.links));
    nodes = g.nodes.map((n, i) => ({
      ...n,
      x: Math.cos((i / g.nodes.length) * 6.283) * 220 + (Math.random() - 0.5) * 40,
      y: Math.sin((i / g.nodes.length) * 6.283) * 220 + (Math.random() - 0.5) * 40,
      vx: 0, vy: 0, r: 5 + Math.sqrt(n.degree || 1) * 2.4,
    }));
    byId = new Map(nodes.map((n) => [n.id, n]));
    links = g.links
      .map((l) => ({ ...l, source: byId.get(l.source), target: byId.get(l.target) }))
      .filter((l) => l.source && l.target);
    buildLegend();
    resize();
    requestAnimationFrame(loop);
  });

  function buildLegend() {
    LAYERS.forEach((layer) => {
      const b = document.createElement("button");
      b.innerHTML = `<span class="dot" style="background:${COLOR[layer]}"></span>${layer}`;
      b.onclick = () => {
        hidden.has(layer) ? hidden.delete(layer) : hidden.add(layer);
        b.classList.toggle("off");
      };
      legend.appendChild(b);
    });
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);

  function visible(n) { return !hidden.has(n.layer); }

  function physics() {
    const vis = nodes.filter(visible);
    for (let i = 0; i < vis.length; i++) {
      const a = vis[i];
      for (let j = i + 1; j < vis.length; j++) {
        const b = vis[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy || 0.01;
        const f = 2600 / d2;
        const d = Math.sqrt(d2);
        const fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
    }
    links.forEach((l) => {
      if (!visible(l.source) || !visible(l.target)) return;
      let dx = l.target.x - l.source.x, dy = l.target.y - l.source.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d - 90) * 0.015;
      const fx = (dx / d) * f, fy = (dy / d) * f;
      l.source.vx += fx; l.source.vy += fy; l.target.vx -= fx; l.target.vy -= fy;
    });
    vis.forEach((n) => {
      if (n === dragging) return;
      n.vx += -n.x * 0.0015; n.vy += -n.y * 0.0015; // gentle centering
      n.x += (n.vx *= 0.86); n.y += (n.vy *= 0.86);
    });
  }

  function fit() {
    const vis = nodes.filter(visible);
    if (!vis.length) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    vis.forEach((n) => { minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x); minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y); });
    const rect = canvas.getBoundingClientRect();
    const pad = 60;
    const s = Math.min((rect.width - pad) / (maxX - minX || 1), (rect.height - pad) / (maxY - minY || 1), 2.2);
    transform = { s, x: rect.width / 2 - ((minX + maxX) / 2) * s, y: rect.height / 2 - ((minY + maxY) / 2) * s };
  }
  const toScreen = (n) => ({ x: n.x * transform.s + transform.x, y: n.y * transform.s + transform.y });
  const toWorld = (px, py) => ({ x: (px - transform.x) / transform.s, y: (py - transform.y) / transform.s });

  function loop() {
    physics(); fit();
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    const focus = selected || hover;
    const near = focus ? new Set([focus.id]) : null;
    if (focus) links.forEach((l) => { if (l.source === focus) near.add(l.target.id); if (l.target === focus) near.add(l.source.id); });

    links.forEach((l) => {
      if (!visible(l.source) || !visible(l.target)) return;
      const a = toScreen(l.source), b = toScreen(l.target);
      const active = focus && (l.source === focus || l.target === focus);
      ctx.strokeStyle = active ? "rgba(94,234,212,0.7)" : "rgba(120,140,190,0.16)";
      ctx.lineWidth = active ? 1.8 : 0.7 + (l.tier || 1) * 0.25;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    });

    nodes.forEach((n) => {
      if (!visible(n)) return;
      const p = toScreen(n);
      const dim = focus && !near.has(n.id);
      ctx.globalAlpha = dim ? 0.18 : 1;
      ctx.beginPath(); ctx.arc(p.x, p.y, n.r, 0, 6.2832);
      ctx.fillStyle = COLOR[n.layer] || COLOR.unknown; ctx.fill();
      if (n === focus) { ctx.lineWidth = 2.5; ctx.strokeStyle = "#fff"; ctx.stroke(); }
      if (n.degree >= 6 || n === focus) {
        ctx.globalAlpha = dim ? 0.25 : 1;
        ctx.fillStyle = "#e8ecf8"; ctx.font = "600 11px Inter, sans-serif";
        ctx.fillText(n.label, p.x + n.r + 3, p.y + 3);
      }
      ctx.globalAlpha = 1;
    });
    requestAnimationFrame(loop);
  }

  function pick(px, py) {
    const w = toWorld(px, py);
    let best = null, bd = 14 / transform.s;
    nodes.forEach((n) => {
      if (!visible(n)) return;
      const d = Math.hypot(n.x - w.x, n.y - w.y);
      if (d < Math.max(n.r + 4, bd) && (!best || d < bd)) { best = n; bd = d; }
    });
    return best;
  }

  function showNode(n) {
    const edges = links.filter((l) => l.source === n || l.target === n);
    const rows = edges.map((l) => {
      const out = l.source === n;
      const other = out ? l.target : l.source;
      return `<div class="row"><span class="rel">${out ? "" : "← "}${l.relation}${out ? " →" : ""}</span> <strong>${other.label}</strong>
        <span class="pill" style="background:${COLOR[other.layer]}">${other.layer}</span>
        <div class="ev">tier ${l.tier} · ${l.evidence || ""}</div></div>`;
    }).join("");
    info.innerHTML = `<h3>${n.label}</h3>
      <span class="pill" style="background:${COLOR[n.layer]}">${n.layer}</span>
      <span class="pill" style="background:#334155;color:#e8ecf8">${n.source}</span>
      <p class="muted">${edges.length} relations</p>${rows}`;
  }

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (dragging) { const w = toWorld(mx, my); dragging.x = w.x; dragging.y = w.y; dragging.vx = dragging.vy = 0; return; }
    hover = pick(mx, my);
    canvas.style.cursor = hover ? "pointer" : "grab";
    if (hover && !selected) showNode(hover);
  });
  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    dragging = pick(e.clientX - rect.left, e.clientY - rect.top);
    if (dragging) { selected = dragging; showNode(dragging); }
  });
  window.addEventListener("mouseup", () => (dragging = null));

  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    if (!q) { selected = null; return; }
    const hit = nodes.find((n) => n.label.toLowerCase().includes(q));
    if (hit) { selected = hit; showNode(hit); }
  });
})();
