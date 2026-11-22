/* VN-Index · Dự báo biến động — ứng dụng minh hoạ luận văn (Behavioral Hybrid XGBoost-LSTM) */
"use strict";

const COLORS = { M0: "#8FA6BF", M1: "#B58CE0", M2: "#4FC1B0", M3: "#8DB8F2", M4: "#4C8FE0", M5: "#F0A35E", M5B: "#E8C35A", y: "#EAF0F7" };
const ORDER = ["M0", "M1", "M2", "M3", "M4", "M5"];
const HZ = [1, 5, 22];
const HLABEL = { 1: "Phiên tới", 5: "Tuần tới", 22: "Tháng tới" };
const HLONG = { 1: "h = 1 (1 phiên)", 5: "h = 5 (1 tuần)", 22: "h = 22 (1 tháng)" };
const DESC = {
  M0: "Mốc đối chứng đơn giản nhất, phổ biến trong quản trị rủi ro (RiskMetrics). Phương sai là trung bình trượt mũ của bình phương lợi suất với hệ số suy giảm λ = 0,94.",
  M1: "Mô hình phương sai có điều kiện chuẩn mực, phân phối Student-t bất đối xứng. Ước lượng lại mỗi 22 phiên trên cửa sổ mở rộng, giữa hai lần ước lượng thì cập nhật đệ quy.",
  M2: "Học máy đơn lẻ dạng cây tăng cường trên toàn bộ 27 biến. Siêu tham số tinh chỉnh bằng tối ưu Bayes trên tập kiểm định năm 2020.",
  M3: "Mạng nơ-ron hồi quy hai tầng (48 và 24 đơn vị) học trên cửa sổ 20 phiên của 19 đặc trưng kỹ thuật, dừng sớm theo tập kiểm định nội bộ, lấy trung bình nhiều hạt giống.",
  M4: "Mô hình lai nối tiếp: XGBoost xếp hạng 19 đặc trưng kỹ thuật, giữ 15 biến quan trọng nhất đưa vào LSTM. Là đối chứng trực tiếp của M5.",
  M5: "Mô hình đề xuất của luận văn. Cùng kiến trúc với M4 nhưng bể ứng viên có thêm 8 biến hành vi (chú ý, bầy đàn, phản ứng thái quá, trú ẩn an toàn). Chênh lệch M5 so với M4 là đóng góp riêng của yếu tố hành vi.",
};
const BEH_GROUP = { vol_ratio: "Sự chú ý", vol_anomaly: "Sự chú ý", return_zscore_20: "Phản ứng thái quá", up_streak: "Bầy đàn", down_streak: "Bầy đàn", vol_x_anom: "Cộng hưởng", gold_return_lag1: "Trú ẩn an toàn", usdvnd_return_lag1: "Trú ẩn an toàn" };

const S = { latest: null, history: null, metrics: null, recent: null, track: null, thesis: null, bt: {},
  todayModel: "M5", todayH: 1, histRange: 252, mdlH: 5, mdlRange: 0, mdlSel: new Set(["M1", "M5"]),
  cmpSrc: "app", cmpPeriod: "thesis", cmpH: 1, dmLoss: "SE", mcsLoss: "SE", dmBase: "M5", impH: 1, impModel: "M5" };
const charts = {};

/* ---------------------------------------------------------------- tiện ích */
const $ = (s) => document.querySelector(s);
const nf = (d) => new Intl.NumberFormat("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = (v, d = 2) => (v == null || !isFinite(v) ? "—" : nf(d).format(v));
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "—" : nf(d).format(v * 100) + "%");
const fdate = (s) => (s ? s.split("-").reverse().join("/") : "—");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const mname = (c) => (S.latest && S.latest.models[c] ? S.latest.models[c].name : (S.thesis && S.thesis.names[c]) || c);
const mshort = (c) => (S.latest && S.latest.models[c] ? S.latest.models[c].short : c === "M5B" ? "M5 + độ rộng" : c);
const flabel = (f) => (S.latest && S.latest.feature_labels[f]) || f;

function dataBase() {
  if (window.APP_DATA_BASE) return window.APP_DATA_BASE;
  if (location.protocol === "file:" && window.APP_REMOTE) return window.APP_REMOTE + "data/";
  return "data/";
}
async function getJSON(name, local) {
  const key = "vnv:" + name;
  const urls = local ? [name] : [dataBase() + name];
  if (!local && location.protocol === "file:") urls.push("data/" + name);   // dữ liệu đóng gói trong APK
  for (let i = 0; i < urls.length; i++) {
    try {
      const u = urls[i].startsWith("http") ? urls[i] + "?t=" + Date.now() : urls[i];
      const r = await fetch(u, { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      const txt = await r.text();
      const js = JSON.parse(txt);
      if (i === 0) { try { localStorage.setItem(key, txt); } catch (e) { /* bộ nhớ đầy */ } }
      else S.offline = true;
      return js;
    } catch (e) {
      if (i === 0) {
        try { const t = localStorage.getItem(key); if (t) { S.offline = true; return JSON.parse(t); } } catch (e2) { /* */ }
      }
    }
  }
  return null;
}
function seg(el, items, cur, onPick) {
  el.innerHTML = items.map(([v, t, dis]) => `<button data-v="${v}" class="${String(v) === String(cur) ? "on" : ""}" ${dis ? "disabled style='opacity:.35'" : ""}>${t}</button>`).join("");
  el.onclick = (e) => {
    const b = e.target.closest("button"); if (!b || b.disabled) return;
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    onPick(b.dataset.v);
  };
}
function chips(el, codes, isOn, onPick, extra) {
  el.innerHTML = codes.map((c) => `<button class="chip ${isOn(c) ? "on" : ""}" style="--c:${COLORS[c]}" data-c="${c}"><i></i>${c} · ${esc(mshort(c))}${extra ? `<small>${extra(c)}</small>` : ""}</button>`).join("");
  el.onclick = (e) => { const b = e.target.closest(".chip"); if (b) onPick(b.dataset.c); };
}
function pctRank(arr, v) {
  if (v == null) return null;
  let n = 0, k = 0;
  for (const x of arr) { if (x == null) continue; n++; if (x <= v) k++; }
  return n ? (k / n) * 100 : null;
}
function level(pr) {
  if (pr == null) return ["—", ""];
  if (pr < 25) return ["Thấp", "l0"];
  if (pr < 75) return ["Bình thường", "l1"];
  if (pr < 90) return ["Cao", "l2"];
  return ["Rất cao", "l3"];
}

Chart.defaults.color = "#8FA6BF";
Chart.defaults.font.family = '"BVP", system-ui, sans-serif';
Chart.defaults.font.size = 11;
Chart.defaults.borderColor = "rgba(34,64,95,.6)";
Chart.defaults.animation = false;
function mkChart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart(document.getElementById(id), cfg); return charts[id]; }
const tipDate = { callbacks: { title: (it) => fdate(it[0].label) } };
const xDates = { ticks: { maxTicksLimit: 7, maxRotation: 0, callback: function (v) { const s = this.getLabelForValue(v); return s ? s.slice(5, 7) + "/" + s.slice(0, 4) : ""; } }, grid: { display: false } };

/* ---------------------------------------------------------------- tải dữ liệu */
async function boot() {
  document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  window.addEventListener("hashchange", () => showTab(location.hash.replace("#", "")));
  const [latest, history, metrics, recent, track, thesis] = await Promise.all([
    getJSON("latest.json"), getJSON("history.json"), getJSON("metrics.json"), getJSON("recent.json"), getJSON("track.json"), getJSON("thesis.json", true)]);
  Object.assign(S, { latest, history, metrics, recent, track, thesis });
  $("#ver").textContent = window.APP_VERSION ? "phiên bản " + window.APP_VERSION : "";
  if (!latest) {
    setStatus("err", "Chưa có dữ liệu");
    document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="card warn"><h3>Chưa tải được dữ liệu</h3><p>Ứng dụng chưa có kết quả chạy đầu tiên hoặc thiết bị đang ngoại tuyến. Hãy thử lại sau ít phút.</p></div>`);
    renderAbout();
    return;
  }
  const last = latest.status.last_session;
  const ageDays = (Date.now() - new Date(last + "T15:00:00+07:00").getTime()) / 864e5;
  setStatus(S.offline ? "stale" : ageDays <= 4 ? "ok" : "stale",
    `Phiên ${fdate(last)} · cập nhật ${latest.status.generated_at.slice(11, 16)} ${fdate(latest.status.generated_at.slice(0, 10))}${S.offline ? " · ngoại tuyến" : ""}`);
  renderToday();
  renderAbout();
  showTab(location.hash.replace("#", "") || "today");
}
function setStatus(cls, txt) { $("#status").className = "status " + cls; $("#statusText").textContent = txt; $("#status").title = txt; }

function showTab(t) {
  if (!document.getElementById("tab-" + t)) t = "today";
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x.id === "tab-" + t));
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  if (location.hash !== "#" + t) history.replaceState(null, "", "#" + t);
  if (!S.latest) return;
  if (t === "model") renderModel();
  if (t === "compare") renderCompare();
  if (t === "features") renderFeatures();
  window.scrollTo({ top: 0 });
}

/* ================================================================ HÔM NAY */
function renderToday() {
  const L = S.latest, m = L.market, H = S.history;
  $("#mDate").textContent = fdate(m.date);
  $("#mClose").textContent = fmt(m.close, 2);
  const up = m.change >= 0;
  $("#mChg").className = "chg " + (up ? "up" : "down");
  $("#mChg").textContent = `${up ? "▲ +" : "▼ "}${fmt(m.change, 2)} điểm (${up ? "+" : ""}${fmt(m.change_pct, 2)}%)`;
  $("#mVol").textContent = pct(m.vol20) + "/năm";
  const [lt, lc] = level(pctRank(H.vol20, m.vol20));
  $("#mVolLvl").className = "lvl " + lc; $("#mVolLvl").textContent = "Mức " + lt.toLowerCase() + " so với lịch sử";

  chips($("#todayModels"), ORDER, (c) => c === S.todayModel, (c) => { S.todayModel = c; renderToday(); });
  const fc = L.forecast || {};
  $("#fcCards").innerHTML = HZ.map((h) => {
    const v = fc[h] ? fc[h][S.todayModel] : null;
    const pr = pctRank(H.vol20, v);
    const [lt2, lc2] = level(pr);
    const band = v != null ? m.close * v * Math.sqrt(h / 252) : null;
    const dlt = v != null && m.vol20 != null ? v - m.vol20 : null;
    const d = fc[h + "_date"] ? fc[h + "_date"][S.todayModel] : null;
    return `<div class="fc" style="--c:${COLORS[S.todayModel]}">
      <div class="h"><span>${HLABEL[h]} · h = ${h}</span><span class="lvl ${lc2}" style="margin:0">${lt2}</span></div>
      <div class="v">${pct(v)} <small>/năm</small></div>
      <div class="gauge"><i style="left:${pr == null ? 50 : pr}%"></i></div>
      <div class="gl"><span>thấp</span><span>bình thường</span><span>cao</span></div>
      <div class="rg">Dao động kỳ vọng: <b>±${fmt(band, 1)} điểm</b>${band != null ? ` (≈ ${fmt(m.close - band, 0)} – ${fmt(m.close + band, 0)})` : ""}</div>
      <div class="delta">${dlt == null ? "" : (dlt >= 0 ? "Cao hơn" : "Thấp hơn") + " biến động 20 phiên hiện tại " + fmt(Math.abs(dlt) * 100, 1) + " điểm %"}${d && d !== m.date ? " · dự báo tại phiên " + fdate(d) : ""}</div>
    </div>`;
  }).join("");

  seg($("#todayH"), HZ.map((h) => [h, "h = " + h]), S.todayH, (v) => { S.todayH = +v; renderTodayBars(); });
  renderTodayBars();
  seg($("#histRange"), [[63, "3T"], [126, "6T"], [252, "1N"], [756, "3N"], [0, "Tất cả"]], S.histRange, (v) => { S.histRange = +v; renderHist(); });
  renderHist();
}
function renderTodayBars() {
  const L = S.latest, f = (L.forecast || {})[S.todayH] || {};
  const vals = ORDER.map((c) => f[c]).filter((x) => x != null);
  const mx = Math.max(...vals, L.market.vol20 || 0) * 1.12 || 1;
  const ref = L.market.vol20;
  $("#todayBars").innerHTML = ORDER.map((c) => `<div class="bar" style="--c:${COLORS[c]}">
      <div class="n"><b>${c}</b> · ${esc(mshort(c))}</div>
      <div class="track"><div class="fill" style="width:${f[c] == null ? 0 : (f[c] / mx) * 100}%"></div>${ref ? `<div class="ref" style="left:${(ref / mx) * 100}%"></div>` : ""}</div>
      <div class="val">${pct(f[c])}</div></div>`).join("");
}
function renderHist() {
  const H = S.history, n = H.dates.length, k = S.histRange ? Math.max(0, n - S.histRange) : 0;
  mkChart("cHist", {
    type: "line",
    data: { labels: H.dates.slice(k), datasets: [
      { label: "VN-Index", data: H.close.slice(k), borderColor: "#8DB8F2", borderWidth: 1.6, pointRadius: 0, yAxisID: "y", tension: 0.1 },
      { label: "Biến động 20 phiên (%/năm)", data: H.vol20.slice(k).map((v) => (v == null ? null : v * 100)), borderColor: "#F0A35E", backgroundColor: "rgba(240,163,94,.12)", fill: true, borderWidth: 1.4, pointRadius: 0, yAxisID: "y1", tension: 0.1 }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { boxWidth: 12 } }, tooltip: { ...tipDate, callbacks: { ...tipDate.callbacks, label: (c) => c.dataset.label + ": " + fmt(c.parsed.y, c.datasetIndex ? 1 : 2) } } },
      scales: { x: xDates, y: { position: "left", ticks: { callback: (v) => fmt(v, 0) } }, y1: { position: "right", grid: { display: false }, ticks: { callback: (v) => v + "%" } } } },
  });
}

/* ================================================================ MÔ HÌNH */
async function loadBT(h) {
  if (!S.bt[h]) S.bt[h] = await getJSON(`backtest_h${h}.json`);
  return S.bt[h];
}
async function renderModel() {
  seg($("#mdlH"), HZ.map((h) => [h, HLONG[h]]), S.mdlH, (v) => { S.mdlH = +v; renderModel(); });
  seg($("#mdlRange"), [[252, "1 năm"], [504, "2 năm"], [0, "Từ 2021"]], S.mdlRange, (v) => { S.mdlRange = +v; drawBT(); });
  const met = S.metrics && S.metrics.all[S.mdlH];
  const rank = {};
  if (met) [...met.table].sort((a, b) => a.RMSE - b.RMSE).forEach((r, i) => (rank[r.model] = i + 1));
  chips($("#mdlModels"), ORDER, (c) => S.mdlSel.has(c), (c) => {
    if (S.mdlSel.has(c)) { if (S.mdlSel.size > 1) S.mdlSel.delete(c); } else S.mdlSel.add(c);
    S.lastPicked = c; renderModel();
  }, (c) => (rank[c] ? "#" + rank[c] : ""));
  await loadBT(S.mdlH);
  drawBT();
  renderModelInfo(S.lastPicked && S.mdlSel.has(S.lastPicked) ? S.lastPicked : [...S.mdlSel].pop());
  renderTrack();
}
function drawBT() {
  const B = S.bt[S.mdlH];
  if (!B) { $("#cBT").parentElement.innerHTML = `<div class="empty">Chưa có kết quả walk-forward cho tầm này.</div><canvas id="cBT"></canvas>`; return; }
  const R = S.recent && S.recent[S.mdlH];
  const lastBT = B.dates[B.dates.length - 1];
  const extra = R ? R.dates.filter((d) => d > lastBT) : [];
  const labels = B.dates.concat(extra);
  const n = labels.length, k = S.mdlRange ? Math.max(0, n - S.mdlRange) : 0;
  const ds = [{ label: "Thực tế", data: B.actual.concat(extra.map(() => null)).slice(k).map((v) => (v == null ? null : v * 100)), borderColor: "rgba(234,240,247,.85)", borderWidth: 1.1, pointRadius: 0, order: 9 }];
  ORDER.filter((c) => S.mdlSel.has(c)).forEach((c) => {
    const p = (B.pred[c] || []).concat(extra.map((d) => (R && R[c] ? R[c][R.dates.indexOf(d)] : null)));
    ds.push({ label: c + " · " + mshort(c), data: p.slice(k).map((v) => (v == null ? null : v * 100)), borderColor: COLORS[c], borderWidth: c === "M5" ? 2 : 1.5, pointRadius: 0, spanGaps: false,
      segment: { borderDash: (ctx) => (labels[k + ctx.p1DataIndex] > lastBT ? [5, 4] : undefined) } });
  });
  mkChart("cBT", { type: "line", data: { labels: labels.slice(k), datasets: ds },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { boxWidth: 12 } }, tooltip: { ...tipDate, callbacks: { ...tipDate.callbacks, label: (c) => c.dataset.label + ": " + fmt(c.parsed.y, 1) + "%" } } },
      scales: { x: xDates, y: { title: { display: true, text: "Biến động (%/năm)" }, ticks: { callback: (v) => v + "%" } } } } });
}
function renderModelInfo(c) {
  const M = S.latest.models[c], met = S.metrics && S.metrics.all[S.mdlH];
  const row = met ? met.table.find((r) => r.model === c) : null;
  let rk = "";
  if (met) {
    const sorted = [...met.table].sort((a, b) => a.RMSE - b.RMSE);
    rk = `Xếp hạng RMSE: <b>${sorted.findIndex((r) => r.model === c) + 1}/${sorted.length}</b>`;
  }
  const lv = S.latest.live_models[S.mdlH] && S.latest.live_models[S.mdlH][c];
  $("#mdlInfo").innerHTML = `<div class="card-h"><h3><span class="tbl"><i class="sw" style="background:${COLORS[c]}"></i></span>${c} · ${esc(M.name)}</h3><span class="pill">${esc(M.group)}</span></div>
    <p class="sub" style="margin-bottom:10px">${DESC[c]}</p>
    <div class="kv">
      <div>Đầu vào</div><div>${esc(M.input)}</div>
      <div>Đánh giá ngoài mẫu</div><div>${met ? fdate(met.start) + " → " + fdate(met.end) + " · N = " + fmt(row ? row.N : null, 0) : "—"}</div>
      <div>RMSE · MAE · QLIKE</div><div>${row ? fmt(row.RMSE, 5) + " · " + fmt(row.MAE, 5) + " · " + fmt(row.QLIKE, 4) : "—"}</div>
      <div>So với các mô hình</div><div>${rk || "—"}</div>
      ${lv ? `<div>Huấn luyện gần nhất</div><div>dữ liệu đến ${fdate(lv.info.trained_until)}, ${fmt(lv.info.n_train, 0)} mẫu học</div>` : ""}
      ${lv && lv.behav_sel && c === "M5" ? `<div>Biến hành vi được chọn</div><div>${lv.behav_sel.length ? lv.behav_sel.map((b) => `<span class="tag bh">${esc(b)}</span>`).join(" ") : "không có"}</div>` : ""}
    </div>`;
}
function renderTrack() {
  const T = (S.track || []).filter((r) => r.h === S.mdlH);
  const dates = [...new Set(T.map((r) => r.date))].sort().reverse().slice(0, 15);
  if (!dates.length) { $("#trackTbl").innerHTML = `<tr><td class="empty">Nhật ký sẽ có từ lần chạy hằng ngày đầu tiên.</td></tr>`; return; }
  const cols = ORDER.filter((c) => S.mdlSel.has(c));
  const get = (d, c) => { const r = T.find((x) => x.date === d && x.model === c); return r ? r : null; };
  $("#trackTbl").innerHTML = `<tr><th>Phiên</th>${cols.map((c) => `<th style="color:${COLORS[c]}">${c}</th>`).join("")}<th>Thực tế</th></tr>` +
    dates.map((d) => { const a = (T.find((x) => x.date === d && x.actual != null) || {}).actual;
      return `<tr><td>${fdate(d)}</td>${cols.map((c) => `<td>${pct((get(d, c) || {}).pred)}</td>`).join("")}<td class="${a == null ? "muted" : ""}">${a == null ? "chờ" : pct(a)}</td></tr>`; }).join("");
}

/* ================================================================ SO SÁNH */
function cmpData() {
  const h = S.cmpH;
  if (S.cmpSrc === "thesis") {
    const T = S.thesis;
    return { thesis: true, start: "2021-01-01", end: "2025-12-31", models: T.t44[h].map((r) => r.model), table: T.t44[h],
      mcs: T.mcs[h], dmPairs: T.t45[h], abl: T.ablation };
  }
  const M = S.metrics && S.metrics[S.cmpPeriod] && S.metrics[S.cmpPeriod][h];
  if (!M) return null;
  return { thesis: false, start: M.start, end: M.end, models: M.table.map((r) => r.model), table: M.table, mcs: M.mcs, dm: M.dm, yearly: M.yearly_rmse };
}
function renderCompare() {
  seg($("#cmpSrc"), [["app", "Ứng dụng chạy lại"], ["thesis", "Số liệu luận văn V2.5"]], S.cmpSrc, (v) => { S.cmpSrc = v; if (v === "thesis") S.dmBase = "M5"; renderCompare(); });
  seg($("#cmpPeriod"), [["thesis", "2021–2025"], ["all", "2021 → nay", S.cmpSrc === "thesis"]], S.cmpSrc === "thesis" ? "thesis" : S.cmpPeriod, (v) => { S.cmpPeriod = v; renderCompare(); });
  seg($("#cmpH"), HZ.map((h) => [h, HLONG[h]]), S.cmpH, (v) => { S.cmpH = +v; renderCompare(); });
  const D = cmpData();
  if (!D) { $("#cmpTbl").innerHTML = `<tr><td class="empty">Chưa có kết quả walk-forward.</td></tr>`; return; }
  $("#cmpRange").textContent = `${fdate(D.start)} → ${fdate(D.end)}`;
  $("#cmpTitle").textContent = D.thesis ? "Bảng 4.4 trong luận văn" : "Chỉ tiêu dự báo ngoài mẫu (ứng dụng tự chạy lại)";
  const cols = D.thesis ? ["RMSE", "MAE", "MAPE", "QLIKE"] : ["RMSE", "MAE", "QLIKE", "N"];
  const best = {};
  cols.forEach((k) => { if (k === "N") return; const v = D.table.map((r) => r[k]).filter((x) => x != null); best[k] = Math.min(...v); });
  const dg = { RMSE: 5, MAE: 5, QLIKE: 4, MAPE: 2, N: 0 };
  const memb = (l, c) => D.mcs && D.mcs[l] && D.mcs[l][c] != null ? (D.mcs[l][c] >= 0.1 ? `<span class="tag yes">có</span>` : `<span class="tag no">không</span>`) : "—";
  $("#cmpTbl").innerHTML = `<tr><th>Mô hình</th>${cols.map((k) => `<th>${k}${k === "MAPE" ? " (%)" : ""}</th>`).join("")}<th>MCS (SE)</th><th>MCS (QLIKE)</th></tr>` +
    D.table.map((r) => `<tr class="${r.model === "M5" ? "me" : ""}"><td><i class="sw" style="background:${COLORS[r.model]}"></i><b>${r.model}</b> · ${esc(mshort(r.model))}</td>${cols.map((k) => `<td class="${best[k] != null && r[k] === best[k] ? "best" : ""}">${r[k] == null ? "—" : fmt(r[k], dg[k])}</td>`).join("")}<td>${memb("SE", r.model)}</td><td>${memb("QLIKE", r.model)}</td></tr>`).join("");
  $("#cmpNote").innerHTML = D.thesis
    ? `Số liệu cố định trích từ luận văn (${esc(S.thesis.version)}), giai đoạn ${esc(S.thesis.period)}. M5B bổ sung 4 biến độ rộng thị trường, chỉ dùng cho phân tích độ vững.`
    : `Ứng dụng chạy lại toàn bộ quy trình với dữ liệu lấy từ nguồn công khai và ${S.latest.status.config.N_SEEDS} hạt giống LSTM, nên số liệu có thể lệch nhẹ so với luận văn. Chỉ so sánh trên các phiên mà mọi mô hình đều có dự báo.`;

  // DM
  seg($("#dmLoss"), [["SE", "SE"], ["QLIKE", "QLIKE"]], S.dmLoss, (v) => { S.dmLoss = v; renderCompare(); });
  const sel = $("#dmBase");
  sel.innerHTML = (D.thesis ? ["M5"] : D.models).map((c) => `<option value="${c}" ${c === S.dmBase ? "selected" : ""}>${c} · ${esc(mshort(c))}</option>`).join("");
  sel.disabled = D.thesis;
  sel.onchange = () => { S.dmBase = sel.value; renderCompare(); };
  let pairs;
  if (D.thesis) pairs = D.dmPairs.map((p) => ({ a: p.a, b: p.b, ...p[S.dmLoss] }));
  else pairs = D.models.filter((c) => c !== S.dmBase).map((c) => ({ a: S.dmBase, b: c, ...(D.dm[S.dmLoss][S.dmBase][c] || {}) }));
  $("#dmBox").innerHTML = pairs.map((p) => {
    const v = Math.max(-6, Math.min(6, p.dm || 0)), w = (Math.abs(v) / 6) * 50;
    const sig = p.p != null && p.p < 0.05;
    const verdict = p.dm == null ? `<span class="tag tie">—</span>` : !sig ? `<span class="tag tie">không khác biệt</span>` : p.dm < 0 ? `<span class="tag win">${p.a} tốt hơn</span>` : `<span class="tag lose">${p.a} kém hơn</span>`;
    return `<div class="dmrow"><div><b>${p.a}</b> vs ${p.b}</div>
      <div class="dmbar"><i style="${v < 0 ? `right:50%;width:${w}%` : `left:50%;width:${w}%`};background:${sig ? (v < 0 ? "var(--up)" : "var(--down)") : "var(--dim)"}"></i></div>
      <div class="p">${fmt(p.dm, 2)} · p ${fmt(p.p, 4)}<br>${verdict}</div></div>`;
  }).join("");

  // MCS
  seg($("#mcsLoss"), [["SE", "SE"], ["QLIKE", "QLIKE"]], S.mcsLoss, (v) => { S.mcsLoss = v; renderCompare(); });
  const mc = (D.mcs && D.mcs[S.mcsLoss]) || {};
  $("#mcsBox").innerHTML = D.models.map((c) => { const p = mc[c];
    return `<div class="mcsrow"><div><b>${c}</b> · ${esc(mshort(c))}</div><div class="t"><i style="width:${p == null ? 0 : p * 100}%;background:${COLORS[c]}"></i></div><div class="p">${fmt(p, 3)}</div><div>${p == null ? "" : p >= 0.1 ? `<span class="tag yes">thuộc</span>` : `<span class="tag no">loại</span>`}</div></div>`; }).join("");

  // Ablation M5 vs M4 qua 3 tầm
  $("#ablBox").innerHTML = HZ.map((h) => {
    let m4, m5, p, src;
    if (S.cmpSrc === "thesis") {
      const a = S.thesis.ablation[h]; m4 = a.M4; m5 = a.M5;
      const pr = S.thesis.t45[h].find((x) => x.a === "M5" && x.b === "M4"); p = pr ? pr.SE.p : null; src = "luận văn";
    } else {
      const M = S.metrics[S.cmpPeriod] && S.metrics[S.cmpPeriod][h]; if (!M) return "";
      const r4 = M.table.find((r) => r.model === "M4"), r5 = M.table.find((r) => r.model === "M5");
      if (!r4 || !r5) return ""; m4 = r4.RMSE; m5 = r5.RMSE; p = (M.dm.SE.M5.M4 || {}).p; src = "ứng dụng";
    }
    const g = ((m4 - m5) / m4) * 100;
    return `<div class="a"><div class="k">${HLONG[h]}</div><div class="g ${g >= 0 ? "pos" : "neg"}">${g >= 0 ? "+" : ""}${fmt(g, 2)}%</div>
      <div class="d">RMSE M4 ${fmt(m4, 5)} → M5 ${fmt(m5, 5)}<br>DM (SE) p = ${fmt(p, 4)} · ${p != null && p < 0.05 ? (g > 0 ? "hành vi giúp cải thiện có ý nghĩa" : "hành vi làm giảm hiệu quả có ý nghĩa") : "chưa có ý nghĩa thống kê"}</div></div>`;
  }).join("");

  // RMSE theo năm
  $("#yearCard").style.display = D.thesis ? "none" : "";
  if (!D.thesis && D.yearly) {
    const yrs = Object.keys(D.yearly);
    mkChart("cYear", { type: "bar", data: { labels: yrs, datasets: D.models.map((c) => ({ label: c, data: yrs.map((y) => D.yearly[y][c]), backgroundColor: COLORS[c], borderRadius: 3 })) },
      options: { maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 12 } }, tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + fmt(c.parsed.y, 5) } } },
        scales: { x: { grid: { display: false } }, y: { title: { display: true, text: "RMSE" } } } } });
  }
}

/* ================================================================ BIẾN SỐ */
function behText(c, v) {
  if (v == null) return ["—", ""];
  switch (c) {
    case "vol_ratio": return ["×" + fmt(v, 2), v > 1.3 ? "Khối lượng cao hơn bình quân 20 phiên, dòng tiền chú ý tăng." : v < 0.8 ? "Khối lượng thấp hơn bình quân, thị trường kém sôi động." : "Khối lượng quanh mức bình quân 20 phiên."];
    case "vol_anomaly": return [v >= 1 ? "Có" : "Không", v >= 1 ? "Khối lượng vượt gấp đôi bình quân, phiên bất thường." : "Không có khối lượng đột biến (ngưỡng ×2)."];
    case "return_zscore_20": return [fmt(v, 2), Math.abs(v) > 2 ? "Lợi suất khác thường so với 20 phiên gần đây, dấu hiệu phản ứng thái quá." : "Lợi suất trong vùng bình thường của 20 phiên gần đây."];
    case "up_streak": return [fmt(v, 0) + " phiên", v >= 3 ? "Chuỗi tăng kéo dài, tâm lý hưng phấn, sợ bỏ lỡ." : v > 0 ? "Thị trường đang có phiên tăng." : "Không trong chuỗi tăng."];
    case "down_streak": return [fmt(v, 0) + " phiên", v >= 3 ? "Chuỗi giảm kéo dài, tâm lý sợ hãi lan truyền." : v > 0 ? "Thị trường đang có phiên giảm." : "Không trong chuỗi giảm."];
    case "vol_x_anom": return [fmt(v, 3), "Biến động 20 phiên nhân tỷ lệ khối lượng: cao khi rủi ro và sự chú ý cùng tăng."];
    case "gold_return_lag1": return [(v >= 0 ? "+" : "") + fmt(v * 100, 2) + "%", v > 0.01 ? "Vàng tăng mạnh phiên trước, tín hiệu e ngại rủi ro." : "Lợi suất vàng thế giới phiên trước."];
    case "usdvnd_return_lag1": return [(v >= 0 ? "+" : "") + fmt(v * 100, 3) + "%", v > 0.002 ? "Tỷ giá tăng nhanh, áp lực dòng vốn." : "Lợi suất tỷ giá USD/VND phiên trước."];
  }
  return [fmt(v, 3), ""];
}
function renderFeatures() {
  const L = S.latest;
  $("#behGrid").innerHTML = L.behav.map((c) => {
    const [v, m] = behText(c, L.behavior_now[c]);
    const p = L.behavior_pct[c];
    return `<div class="beh"><div class="grp">${BEH_GROUP[c]}</div><div class="k">${esc(flabel(c))}</div><div class="code">${c}</div>
      <div class="v">${v}</div><div class="pb"><i style="width:${p == null ? 0 : p}%"></i></div><div class="lbl">${c === "vol_anomaly" ? "xảy ra ở " + fmt(p, 1) + "% số phiên từ 2015" : "phân vị " + fmt(p, 0)}</div><div class="m">${m}</div></div>`;
  }).join("");
  seg($("#impH"), HZ.map((h) => [h, "h = " + h]), S.impH, (v) => { S.impH = +v; renderFeatures(); });
  seg($("#impModel"), [["M5", "M5 · hành vi"], ["M4", "M4 · không hành vi"], ["M2", "M2 · XGBoost"]], S.impModel, (v) => { S.impModel = v; renderFeatures(); });
  const lv = (L.live_models[S.impH] || {})[S.impModel];
  if (!lv || !lv.importance) { $("#impBox").innerHTML = `<div class="empty">Chưa có mô hình huấn luyện.</div>`; }
  else {
    const ent = Object.entries(lv.importance), mx = ent.length ? ent[0][1] : 1, hy = S.impModel !== "M2";
    $("#impBox").innerHTML = ent.map(([f, v], i) => `${hy && i === lv.sel.length ? `<div class="cut">— các biến dưới đây không được chọn vào LSTM —</div>` : ""}<div class="r ${!hy || lv.sel.includes(f) ? "sel" : ""} ${L.behav.includes(f) ? "bv" : ""}"><div class="n" title="${esc(f)}">${esc(flabel(f))}</div><div class="t"><i style="width:${(v / mx) * 100}%"></i></div><div class="val">${fmt(v * 100, 1)}%</div></div>`).join("");
  }
  const T = S.thesis.behav_selected;
  $("#selTbl").innerHTML = `<tr><th>Tầm</th><th style="text-align:left">Ứng dụng (lần huấn luyện gần nhất)</th><th style="text-align:left">Luận văn (Bảng 4.8)</th></tr>` +
    HZ.map((h) => { const a = ((L.live_models[h] || {}).M5 || {}).behav_sel || [];
      const tg = (arr) => arr.length ? arr.map((b) => `<span class="tag bh">${b}</span>`).join(" ") : `<span class="tag no">không</span>`;
      return `<tr><td>${HLONG[h]}</td><td style="text-align:left;white-space:normal">${a.length}/8 · ${tg(a)}</td><td style="text-align:left;white-space:normal">${T[h].length}/8 · ${tg(T[h])}</td></tr>`; }).join("");
}

/* ================================================================ GIỚI THIỆU */
function renderAbout() {
  const T = S.thesis ? S.thesis.meta : {};
  $("#aboutBox").innerHTML = `<div class="lbl">Ứng dụng minh hoạ luận văn thạc sĩ</div><h2>${esc(T.title || "")}</h2>
    <div class="meta"><div>Mô hình đề xuất</div><div><b>${esc(T.model || "")}</b></div><div>Học viên</div><div>${esc(T.author || "")}</div>
    <div>Chuyên ngành</div><div>${esc(T.program || "")}</div><div>Cơ sở đào tạo</div><div>${esc(T.school || "")}</div><div>Người hướng dẫn</div><div>${esc(T.supervisor || "")}</div></div>
    <p class="sub">Câu hỏi nghiên cứu: các tín hiệu hành vi nội sinh trích xuất từ dữ liệu giao dịch có giúp mô hình lai XGBoost-LSTM dự báo biến động VN-Index tốt hơn không, và hiệu quả đó thay đổi thế nào theo tầm dự báo 1, 5 và 22 phiên.</p>
    <div class="steps">
      <div><b>Thu thập dữ liệu</b>VN-Index ngày (OHLCV), giá vàng thế giới, tỷ giá USD/VND từ 2015.</div>
      <div><b>Xây dựng biến</b>Biến động thực hiện nhìn tới trước h phiên; 19 đặc trưng kỹ thuật và 8 biến hành vi.</div>
      <div><b>Chia dữ liệu theo thời gian</b>Huấn luyện đến 2019, tinh chỉnh năm 2020, ngoài mẫu từ 2021, khoảng đệm 22 phiên.</div>
      <div><b>Huấn luyện sáu mô hình</b>Walk-forward cửa sổ mở rộng, tái huấn luyện mỗi 126 phiên.</div>
      <div><b>Đánh giá</b>RMSE, MAE, QLIKE; Diebold-Mariano (HLN); Model Confidence Set.</div>
      <div><b>Phân tích</b>Đóng góp của biến hành vi qua cặp M5 và M4 ở từng tầm dự báo.</div>
    </div>
    <h3>Ứng dụng vận hành thế nào</h3>
    <p class="sub">Sau mỗi phiên giao dịch (khoảng 16 giờ 30 từ thứ Hai đến thứ Sáu), máy chủ GitHub tự lấy dữ liệu mới, tính lại đặc trưng và chạy dự báo cho cả sáu mô hình. Cuối tuần, các mô hình học máy được huấn luyện lại trên toàn bộ dữ liệu, đồng thời cập nhật kết quả walk-forward và các kiểm định so sánh.</p>`;
  const L = S.latest;
  if (!L) { $("#runBox").innerHTML = `<div class="empty">Chưa có lần chạy nào.</div>`; return; }
  const st = L.status, f = st.fetch || {};
  const tr = ((L.live_models[1] || {}).M5 || {}).info || {};
  $("#runBox").innerHTML = `<div class="kv">
    <div>Phiên gần nhất</div><div>${fdate(st.last_session)} (${fmt(st.n_sessions, 0)} phiên từ ${fdate(st.data_start)})</div>
    <div>Lần chạy gần nhất</div><div>${esc(st.generated_at.replace("T", " "))}</div>
    <div>Nguồn VN-Index</div><div>${esc(f.vn_source || "—")}</div>
    <div>Nguồn giá vàng</div><div>${esc(f.gold_source || "—")}</div>
    <div>Nguồn tỷ giá USD/VND</div><div>${esc(f.usdvnd_source || "—")}</div>
    <div>Huấn luyện lại gần nhất</div><div>${tr.trained_at ? esc(tr.trained_at.replace("T", " ")) + " · dữ liệu đến " + fdate(tr.trained_until) : "—"}</div>
    <div>Cấu hình</div><div>walk-forward từ ${fdate(st.config.WF_OOS_START)}, tái huấn luyện ${st.config.WF_REFIT} phiên, đệm ${st.config.EMBARGO} phiên, cửa sổ LSTM ${st.config.LOOKBACK}, top-K ${st.config.TOP_K_FEATURES}, ${st.config.N_SEEDS} hạt giống</div>
    ${f.warnings && f.warnings.length ? `<div>Cảnh báo</div><div>${f.warnings.map(esc).join("<br>")}</div>` : ""}
  </div>`;
}

if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}
boot();
