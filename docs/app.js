/* ==========================================================
   Mushroom Classifier: page logic
   Uses window.MODEL (from model.js, created by export_model.py):
     MODEL.columns   the 21 feature names, in order
     MODEL.features  readable names and answer options for each feature
     MODEL.tree      the trained Decision Tree as a list of nodes
     MODEL.knn       the training mushrooms that KNN compares against
     MODEL.examples  the test mushrooms (never seen during training)
     MODEL.results   accuracy, confusion matrices and more
   ========================================================== */

const M = window.MODEL;
const COLS = M.columns;
const LABELS = ["edible", "poisonous"];

// Questions are grouped by part of the mushroom so the form is easier to fill in
const GROUPS = [
  { title: "Smell & spores", desc: "The strongest clues: start here", cols: ["odor", "spore_print_color", "bruises"] },
  { title: "Cap", desc: "The umbrella-shaped top", cols: ["cap_shape", "cap_surface", "cap_color"] },
  { title: "Gills", desc: "The thin plates under the cap", cols: ["gill_attachment", "gill_spacing", "gill_size", "gill_color"] },
  { title: "Stalk", desc: "The stem", cols: ["stalk_shape", "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring", "stalk_color_above_ring", "stalk_color_below_ring"] },
  { title: "Veil & ring", desc: "The skirt-like ring on the stalk", cols: ["veil_color", "ring_number", "ring_type"] },
  { title: "Where it grows", desc: "Its surroundings", cols: ["population", "habitat"] },
];
const KEY_CLUES = new Set(["odor", "spore_print_color", "stalk_root", "stalk_surface_below_ring"]);
const SWATCHES = {
  Black: "#222", Brown: "#8b5a2b", Buff: "#e3cfa8", Chocolate: "#5a3618", Cinnamon: "#a5552a", Green: "#4f9d55",
  Grey: "#9a9a9a", Orange: "#ee8a2a", Pink: "#f2a7b9", Purple: "#8a5aa6", Red: "#c62f2f", White: "#ffffff", Yellow: "#f2c94c",
};

// Stored mushrooms are 21-letter strings; turn each into a list of codes ("?" means "missing")
const decode = (s) => [...s].map((c) => (c === "?" ? "missing" : c));
const TRAIN = M.knn.rows.map(decode);
const EXAMPLES = M.examples.rows.map(decode);

const $ = (sel) => document.querySelector(sel);
const fmt = (n) => n.toLocaleString("en-US");
const pct = (x) => (x === 1 ? "100%" : (x * 100).toFixed(2) + "%");
const featureLabel = (col) => M.features[col].label;
const optionLabel = (col, code) => M.features[col].options[code] ?? code;

let answers = {};      // e.g. { odor: "n", cap_shape: "x" }
let truth = null;      // true label when a dataset example is loaded

/* ---------- Build the form ---------- */
function buildForm() {
  const host = $("#fieldsets");
  host.innerHTML = GROUPS.map((g) => `
    <fieldset class="group">
      <legend><span class="group-title">${g.title}</span><span class="group-desc">${g.desc}</span></legend>
      ${g.cols.map(questionHTML).join("")}
    </fieldset>`).join("");
  host.addEventListener("change", (e) => {
    if (e.target.type !== "radio") return;
    answers[e.target.name] = e.target.value;
    if (truth) { truth = null; }   // the mushroom no longer matches the dataset example
    update();
  });
}

function questionHTML(col) {
  const opts = Object.entries(M.features[col].options)
    .sort((a, b) => (a[0] === "missing") - (b[0] === "missing") || a[1].localeCompare(b[1]));
  const isColour = col.includes("color");
  return `
    <div class="question" id="q-${col}" role="radiogroup" aria-labelledby="l-${col}">
      <div class="question-label" id="l-${col}">${featureLabel(col)}
        ${KEY_CLUES.has(col) ? '<span class="tag">Key clue</span>' : ""}
        <span class="tag next-tag" hidden>Asked next</span>
      </div>
      <div class="chips">
        ${opts.map(([code, name]) => `
          <label class="chip">
            <input type="radio" name="${col}" value="${code}">
            <span>${isColour && SWATCHES[name] ? `<i class="swatch" style="background:${SWATCHES[name]}"></i>` : ""}${name}</span>
          </label>`).join("")}
      </div>
    </div>`;
}

function setForm(values) {
  answers = { ...values };
  document.querySelectorAll('#fieldsets input[type="radio"]').forEach((r) => { r.checked = answers[r.name] === r.value; });
}

/* ---------- Model 1: Decision Tree ---------- */
// Walk from the top question down. Each node asks "is <col> equal to <val>?"
function runTree(ans) {
  const steps = [];
  let node = M.tree[0];
  while (node.col !== undefined) {
    if (ans[node.col] === undefined) return { done: false, steps, need: node.col };
    const isMatch = ans[node.col] === node.val;
    steps.push({ col: node.col, val: node.val, isMatch });
    node = M.tree[isMatch ? node.yes : node.no];
  }
  const [e, p] = node.samples;
  const label = p > e ? 1 : 0;
  return { done: true, steps, label, samples: node.samples, confidence: Math.max(e, p) / (e + p) };
}

/* ---------- Model 2: K-Nearest Neighbors ---------- */
// Similarity = number of answered features that match a training mushroom.
// (On one-hot switches this ranks mushrooms exactly like the distance scikit-learn uses.)
function runKNN(ans) {
  const idx = COLS.map((c, i) => (ans[c] !== undefined ? i : -1)).filter((i) => i >= 0);
  if (!idx.length) return null;
  const want = idx.map((i) => ans[COLS[i]]);
  const scored = TRAIN.map((row, r) => {
    let m = 0;
    for (let j = 0; j < idx.length; j++) if (row[idx[j]] === want[j]) m++;
    return { r, m };
  });
  scored.sort((a, b) => b.m - a.m || a.r - b.r);
  const best = scored[0].m;
  const tied = scored.filter((s) => s.m === best);   // all mushrooms that are equally (most) similar
  const poisonShare = tied.filter((s) => M.knn.labels[s.r] === 1).length / tied.length;
  const label = poisonShare > 0.5 ? 1 : poisonShare < 0.5 ? 0 : null;
  return { answered: idx.length, best, tied: tied.length, poisonShare, label, top: scored.slice(0, 5) };
}

/* ---------- Render everything after each change ---------- */
function update() {
  const n = Object.keys(answers).length;
  $("#progress-fill").style.width = (n / COLS.length) * 100 + "%";
  $("#progress-text").textContent = `${n} of ${COLS.length} answered`;

  const dt = runTree(answers);
  const knn = runKNN(answers);
  renderTree(dt);
  renderKNN(knn);
  renderVerdict(n, dt, knn);
  renderTruth(dt, knn);

  document.querySelectorAll(".next-tag").forEach((t) => (t.hidden = true));
  if (!dt.done && n > 0) $(`#q-${dt.need} .next-tag`).hidden = false;
}

function setPill(el, label, text) {
  el.dataset.v = label;
  el.textContent = text;
}

function renderTree(dt) {
  const pill = $("#dt-pill");
  const steps = dt.steps.map((s) => `
    <li>${featureLabel(s.col)} is "${optionLabel(s.col, s.val)}"?
      <span class="ans ${s.isMatch ? "ans-yes" : "ans-no"}">${s.isMatch ? "Yes" : "No"}</span></li>`).join("");
  if (dt.done) {
    const name = LABELS[dt.label];
    const [e, p] = dt.samples;
    setPill(pill, name, name === "edible" ? "Edible" : "Poisonous");
    $("#dt-path").innerHTML = steps + `
      <li class="leaf ${name}"><strong>Answer: ${name}</strong>. In training, ${fmt(e + p)} mushrooms ended here:
        ${fmt(e)} edible, ${fmt(p)} poisonous (${Math.round(dt.confidence * 100)}% sure).</li>`;
  } else {
    setPill(pill, "waiting", Object.keys(answers).length ? "Needs more info" : "–");
    $("#dt-path").innerHTML = steps + `
      <li class="need">Next question: <button type="button" class="link-btn" data-goto="${dt.need}">${featureLabel(dt.need)}</button></li>`;
  }
}

function renderKNN(knn) {
  $("#knn-k").textContent = `(k = ${M.knn.k})`;
  const pill = $("#knn-pill");
  if (!knn) {
    setPill(pill, "", "–");
    $("#knn-body").innerHTML = `<p class="knn-summary muted">Answer any question to see the most similar mushrooms.</p>`;
    return;
  }
  const name = knn.label === null ? "waiting" : LABELS[knn.label];
  setPill(pill, name, knn.label === null ? "Split 50/50" : name === "edible" ? "Edible" : "Poisonous");
  const tiedText = knn.tied === 1
    ? `The most similar mushroom matches <strong>${knn.best} of your ${knn.answered}</strong> answers.`
    : `<strong>${fmt(knn.tied)}</strong> mushrooms are equally similar (each matches <strong>${knn.best} of your ${knn.answered}</strong> answers). ${Math.round(knn.poisonShare * 100)}% of them are poisonous.`;
  const rows = knn.top.map(({ r, m }) => {
    const lab = LABELS[M.knn.labels[r]];
    return `<li><span class="nb-label ${lab}">${lab === "edible" ? "Edible" : "Poisonous"}</span>
      <span class="meter"><span class="${lab}" style="width:${(m / knn.answered) * 100}%"></span></span>
      <span class="nb-num">${m}/${knn.answered}</span></li>`;
  }).join("");
  const unusual = knn.answered === COLS.length && knn.best < COLS.length - 2
    ? `<p class="warn-line">No mushroom in the dataset closely matches this combination, so treat both predictions with extra caution.</p>` : "";
  $("#knn-body").innerHTML = `<p class="knn-summary">${tiedText}</p>
    <p class="model-hint">5 closest training mushrooms (answers matched):</p><ul class="neighbours">${rows}</ul>${unusual}`;
}

function renderVerdict(n, dt, knn) {
  const box = $("#verdict");
  let state, kicker, title, sub;
  if (n === 0) {
    state = "empty"; kicker = "Waiting for answers"; title = "Describe your mushroom";
    sub = "Start with its smell. That one answer already tells the models a lot.";
  } else if (!dt.done) {
    state = "unsure"; kicker = `${n} of ${COLS.length} answered`; title = "Keep going…";
    sub = `The Decision Tree needs to know: ${featureLabel(dt.need)}.`;
  } else {
    const agree = knn && knn.label === dt.label;
    const unusual = knn && knn.answered === COLS.length && knn.best < COLS.length - 2;
    if (!agree) {
      state = "unsure"; kicker = "The models disagree"; title = "Treat as poisonous";
      sub = n < COLS.length ? "Answer the remaining questions so KNN can compare more fairly." : "When in doubt, throw it out.";
    } else {
      state = LABELS[dt.label];
      kicker = unusual ? "Low reliability: unusual combination" : n < COLS.length ? `Both models agree · ${n} of ${COLS.length} answered` : "Both models agree";
      title = state === "edible" ? "Likely edible" : "Likely poisonous";
      sub = n < COLS.length ? "Fill in the remaining questions to double-check." : "Remember: this is a learning project, not advice.";
    }
  }
  box.dataset.state = state;
  $("#verdict-kicker").textContent = kicker;
  $("#verdict-title").textContent = title;
  $("#verdict-sub").textContent = sub;
  const bar = $("#mobile-bar");
  bar.dataset.state = state;
  $("#mobile-text").textContent = title;
}

function renderTruth(dt, knn) {
  const el = $("#truth");
  if (truth === null) { el.hidden = true; return; }
  const tick = (ok) => (ok ? "✓ correct" : "✗ wrong");
  el.hidden = false;
  el.innerHTML = `<strong>Real test mushroom</strong> (the models never saw it while training).
    True answer: <strong class="${truth ? "t-poison" : "t-edible"}">${truth ? "Poisonous" : "Edible"}</strong>.
    Decision Tree ${tick(dt.done && dt.label === truth)} · KNN ${tick(knn && knn.label === truth)}`;
}

/* ---------- Buttons ---------- */
function loadExample(which) {
  const pool = EXAMPLES.map((row, i) => i).filter((i) => which === "any" || M.examples.labels[i] === Number(which));
  const i = pool[Math.floor(Math.random() * pool.length)];
  setForm(Object.fromEntries(COLS.map((c, j) => [c, EXAMPLES[i][j]])));
  truth = M.examples.labels[i];
  update();
  if (window.matchMedia("(max-width: 980px)").matches) $("#result").scrollIntoView({ behavior: "smooth" });
}

document.addEventListener("click", (e) => {
  const ex = e.target.closest("[data-example]");
  if (ex) loadExample(ex.dataset.example);
  const go = e.target.closest("[data-goto]");
  if (go) {
    const q = $(`#q-${go.dataset.goto}`);
    q.scrollIntoView({ behavior: "smooth", block: "center" });
    q.classList.remove("is-next"); void q.offsetWidth; q.classList.add("is-next");
    q.querySelector("input").focus({ preventScroll: true });
  }
});
$("#reset").addEventListener("click", () => { setForm({}); truth = null; update(); });

/* ---------- Static sections: odour chart and results ---------- */
function renderOdorChart() {
  const counts = {};
  const all = [...TRAIN.map((r, i) => [r, M.knn.labels[i]]), ...EXAMPLES.map((r, i) => [r, M.examples.labels[i]])];
  const o = COLS.indexOf("odor");
  for (const [row, lab] of all) {
    counts[row[o]] ??= [0, 0];
    counts[row[o]][lab]++;
  }
  const max = Math.max(...Object.values(counts).map(([e, p]) => e + p));
  const rows = Object.entries(counts).sort((a, b) => b[1][0] - a[1][0] || b[1][1] - a[1][1]).map(([code, [e, p]]) => `
    <div class="odor-row">
      <span>${optionLabel("odor", code)}</span>
      <span class="odor-bar" style="width:${((e + p) / max) * 100}%">
        <span class="e" style="flex:${e}"></span><span class="p" style="flex:${p}"></span></span>
      <span class="odor-nums">${fmt(e)} edible · ${fmt(p)} poisonous</span>
    </div>`).join("");
  $("#odor-chart").innerHTML = `<div class="legend"><span><i style="background:var(--edible)"></i>Edible</span><span><i style="background:var(--poison)"></i>Poisonous</span></div>${rows}`;
}

function renderResults() {
  const R = M.results;
  $("#stat-total").textContent = $("#hero-total").textContent = fmt(R.n_total);
  $("#stat-acc").textContent = pct(Math.min(R.dt.test_acc, R.knn.test_acc));
  $("#n-test").textContent = fmt(R.n_test);

  const metrics = [
    ["Decision Tree · test accuracy", pct(R.dt.test_acc), `${fmt(Math.round(R.dt.test_acc * R.n_test))} of ${fmt(R.n_test)} correct`],
    [`KNN (k = ${M.knn.k}) · test accuracy`, pct(R.knn.test_acc), `${fmt(Math.round(R.knn.test_acc * R.n_test))} of ${fmt(R.n_test)} correct`],
    ["Decision Tree · cross-validation", pct(R.dt.cv_acc), "Average of 5 mini-exams on training data"],
    ["KNN · cross-validation", pct(R.knn.cv_acc), "Average of 5 mini-exams on training data"],
  ];
  $("#metric-grid").innerHTML = metrics.map(([l, v, n]) => `
    <div class="metric"><div class="metric-label">${l}</div><div class="metric-value">${v}</div><p class="metric-note">${n}</p></div>`).join("");

  const cm = (title, m) => `
    <div class="cm"><h4>${title}</h4>
      <table>
        <tr><th></th><th>Said edible</th><th>Said poisonous</th></tr>
        <tr><th>Really edible</th><td class="ok">${fmt(m[0][0])}</td><td class="bad ${m[0][1] ? "has" : ""}">${fmt(m[0][1])}</td></tr>
        <tr><th>Really poisonous</th><td class="bad danger ${m[1][0] ? "has" : ""}" title="The dangerous mistake">${fmt(m[1][0])}</td><td class="ok">${fmt(m[1][1])}</td></tr>
      </table></div>`;
  $("#cm-row").innerHTML = cm("Decision Tree", R.dt.cm) + cm(`KNN (k = ${M.knn.k})`, R.knn.cm);

  const maxImp = R.importances[0].value;
  $("#importance").innerHTML = R.importances.filter((f) => f.value >= 0.005).map((f) => `
    <div class="bar-row"><span class="bar-name" title="${f.feature}">${f.feature}</span>
      <span class="bar-track"><span style="width:${(f.value / maxImp) * 100}%"></span></span>
      <span class="bar-val">${Math.round(f.value * 100)}%</span></div>`).join("");

  const winner = R.knn.cv_acc > R.dt.cv_acc ? "KNN is ahead by a hair in cross-validation" : "the Decision Tree is ahead in cross-validation";
  $("#verdict-text").innerHTML = `Both models scored <strong>${pct(R.dt.test_acc)}</strong> and <strong>${pct(R.knn.test_acc)}</strong> on the test mushrooms,
    and neither predicted a poisonous mushroom as edible. On accuracy ${winner} (${pct(R.knn.cv_acc)} vs ${pct(R.dt.cv_acc)}), so in practice they are tied.
    The <strong>Decision Tree</strong> is the better choice for real use: you can read every step of its reasoning, and it's much faster.
    Even a tree that asks only 3 questions gets <strong>${pct(R.small_tree_acc)}</strong> right.`;
}

/* ---------- Theme toggle (remembered in this browser only) ---------- */
function initTheme() {
  const root = document.documentElement;
  try { const saved = localStorage.getItem("theme"); if (saved) root.dataset.theme = saved; } catch (e) { /* storage unavailable */ }
  $("#theme-toggle").addEventListener("click", () => {
    const isDark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = isDark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
  });
}

/* ---------- Mobile: show the verdict bar only while the form is on screen and the result panel is not ---------- */
function initMobileBar() {
  if (!("IntersectionObserver" in window)) return;
  const seen = { form: false, result: false };
  const bar = $("#mobile-bar");
  new IntersectionObserver((entries) => {
    entries.forEach((e) => { seen[e.target.id === "result" ? "result" : "form"] = e.isIntersecting; });
    bar.classList.toggle("show", seen.form && !seen.result);
  }).observe($("#mushroom-form"));
  new IntersectionObserver((entries) => {
    entries.forEach((e) => { seen.result = e.isIntersecting; });
    bar.classList.toggle("show", seen.form && !seen.result);
  }).observe($("#result"));
}

initTheme();
initMobileBar();
buildForm();
renderOdorChart();
renderResults();
update();
