/* Word2Vec Lab — everything runs client-side.
 * Model contract:
 *   model/meta.json   {dim, count, words[], corpus, params{}, trained, quantization}
 *   model/vectors.bin Int8, row-major, count*dim bytes; rows ≈ L2-normalised * 127
 */
(function () {
  'use strict';

  var K = 10;             // neighbours per query
  var MAX_QUERIES = 5;    // clusters that can be compared at once
  var EXAMPLES = ['king', 'paris', 'computer', 'music', 'river', 'doctor', 'happy', 'football'];
  var ANALOGIES = [
    ['man', 'king', 'woman'],
    ['france', 'paris', 'italy'],
    ['big', 'bigger', 'small'],
    ['walk', 'walking', 'swim'],
    ['good', 'better', 'bad']
  ];

  var $ = function (id) { return document.getElementById(id); };

  // ---------- state ----------
  var meta = null, vocab = null, index = null, V = null, N = 0, D = 0;
  var queries = [];       // array of word strings (each has its own colour slot)
  var active = 0;         // which query's neighbours are listed
  var cache = {};         // word -> neighbours [{w, i, s}]
  var lastPoints = null;

  // ---------- theme ----------
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function seriesColor(k) { return cssVar('--s' + ((k % 5) + 1)); }
  function setTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    $('themeToggle').setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    try { localStorage.setItem('w2v-theme', t); } catch (e) {}
    if (lastPoints) { renderNeighbors(); renderPlots(); }
  }
  $('themeToggle').addEventListener('click', function () {
    setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });
  setTheme(document.documentElement.getAttribute('data-theme') || 'dark');

  // ---------- loading ----------
  function setProgress(frac, sub) {
    var pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
    $('progressBar').style.width = pct + '%';
    $('progress').setAttribute('aria-valuenow', String(pct));
    if (sub) $('loaderSub').textContent = sub;
  }
  function fmtMB(b) { return (b / 1048576).toFixed(1) + ' MB'; }

  function fail(err) {
    console.warn(err);
    $('loader').querySelector('.loader').classList.add('error');
    $('loaderTitle').textContent = 'Could not load the model';
    $('loaderSub').textContent = (err && err.message ? err.message : String(err)) +
      (location.protocol === 'file:' ? ' — open this page through a web server (file:// URLs cannot fetch the model).' : '');
  }

  async function fetchBinary(url, expected) {
    var res = await fetch(url);
    if (!res.ok) throw new Error('vectors.bin: HTTP ' + res.status);
    var total = expected || Number(res.headers.get('content-length')) || 0;
    if (!res.body || !res.body.getReader) {
      var b = await res.arrayBuffer();
      setProgress(1);
      return b;
    }
    var reader = res.body.getReader();
    var out = new Uint8Array(total || 0), got = 0, chunks = [];
    for (;;) {
      var r = await reader.read();
      if (r.done) break;
      var c = r.value;
      if (total && got + c.length <= out.length) out.set(c, got); else chunks.push(c);
      got += c.length;
      if (total) setProgress(0.05 + 0.93 * got / total, 'Downloading vectors · ' + fmtMB(got) + ' of ' + fmtMB(total));
    }
    if (chunks.length || got !== out.length) { // size mismatch or unknown: concatenate everything
      var all = new Uint8Array(got), off = 0;
      if (total) { var first = Math.min(out.length, got); all.set(out.subarray(0, first)); off = first; }
      for (var j = 0; j < chunks.length; j++) { all.set(chunks[j], off); off += chunks[j].length; }
      return all.buffer;
    }
    return out.buffer;
  }

  async function load() {
    setProgress(0.02, 'Fetching vocabulary…');
    var res = await fetch('model/meta.json');
    if (!res.ok) throw new Error('meta.json: HTTP ' + res.status);
    meta = await res.json();
    D = meta.dim; vocab = meta.words; N = meta.count || vocab.length;
    if (!D || !vocab || vocab.length !== N) throw new Error('meta.json is malformed (dim/count/words disagree)');
    setProgress(0.05, 'Vocabulary: ' + N.toLocaleString() + ' words · downloading vectors…');

    var buf = await fetchBinary('model/vectors.bin', N * D);
    if (buf.byteLength !== N * D) throw new Error('vectors.bin has ' + buf.byteLength + ' bytes, expected ' + (N * D));
    $('loaderSub').textContent = 'Preparing vectors…';

    var q = new Int8Array(buf);
    V = new Float32Array(N * D);
    for (var r = 0; r < N; r++) {
      var o = r * D, ss = 0, d;
      for (d = 0; d < D; d++) { var x = q[o + d]; ss += x * x; }
      var inv = ss > 0 ? 1 / Math.sqrt(ss) : 0;
      for (d = 0; d < D; d++) V[o + d] = q[o + d] * inv;
    }
    index = new Map();
    for (var i = 0; i < N; i++) index.set(vocab[i], i);
    setProgress(1, 'Ready');
    window.__w2v = { N: N, D: D, bytes: buf.byteLength };
  }

  // ---------- vector maths ----------
  function row(i) { return V.subarray(i * D, i * D + D); }

  // top-k by dot product against a unit vector v; exclude: Set of row indices
  function topK(v, k, exclude) {
    var bestI = [], bestS = [];
    for (var r = 0; r < N; r++) {
      if (exclude && exclude.has(r)) continue;
      var o = r * D, s = 0;
      for (var d = 0; d < D; d++) s += V[o + d] * v[d];
      if (bestS.length < k || s > bestS[bestS.length - 1]) {
        var p = bestS.length < k ? bestS.length : k - 1;
        while (p > 0 && bestS[p - 1] < s) { if (p < k) { bestS[p] = bestS[p - 1]; bestI[p] = bestI[p - 1]; } p--; }
        bestS[p] = s; bestI[p] = r;
        if (bestS.length > k) { bestS.length = k; bestI.length = k; }
      }
    }
    return bestI.map(function (i, j) { return { w: vocab[i], i: i, s: bestS[j] }; });
  }

  function neighbors(word) {
    if (!cache[word]) { var i = index.get(word); cache[word] = topK(row(i), K, new Set([i])); }
    return cache[word];
  }

  function dot(a, b) { var s = 0; for (var d = 0; d < D; d++) s += a[d] * b[d]; return s; }

  // ---------- PCA (center → Gram matrix → Jacobi eigendecomposition) ----------
  // For n points (n ≪ dim) the eigenvalues of X·Xᵀ equal those of the covariance XᵀX
  // (up to 1/(n-1)), and the PCA scores are sqrt(λ_k)·u_k.
  function jacobiEigen(A, n) {
    var Vv = [], i, j, k;
    for (i = 0; i < n; i++) { Vv.push(new Float64Array(n)); Vv[i][i] = 1; }
    for (var sweep = 0; sweep < 100; sweep++) {
      var off = 0;
      for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-20) break;
      for (var p = 0; p < n; p++) {
        for (var q = p + 1; q < n; q++) {
          var apq = A[p][q];
          if (Math.abs(apq) < 1e-15) continue;
          var theta = (A[q][q] - A[p][p]) / (2 * apq);
          var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (k = 0; k < n; k++) { var akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
          for (k = 0; k < n; k++) { var apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
          for (k = 0; k < n; k++) { var vkp = Vv[k][p], vkq = Vv[k][q]; Vv[k][p] = c * vkp - s * vkq; Vv[k][q] = s * vkp + c * vkq; }
        }
      }
    }
    var order = [];
    for (i = 0; i < n; i++) order.push(i);
    order.sort(function (a, b) { return A[b][b] - A[a][a]; });
    return order.map(function (col) {
      var vec = new Float64Array(n);
      for (k = 0; k < n; k++) vec[k] = Vv[k][col];
      return { value: Math.max(0, A[col][col]), vector: vec };
    });
  }

  // rows: array of Float32Array(D). Returns {coords: [[pc1,pc2,pc3]...], ratio:[r1,r2,r3]}
  function pca(rows, nComp, anchor) {
    var n = rows.length, d, i, j;
    var mean = new Float64Array(D);
    for (i = 0; i < n; i++) for (d = 0; d < D; d++) mean[d] += rows[i][d] / n;
    var X = rows.map(function (r) { var x = new Float64Array(D); for (var d2 = 0; d2 < D; d2++) x[d2] = r[d2] - mean[d2]; return x; });
    var G = [], trace = 0;
    for (i = 0; i < n; i++) G.push(new Float64Array(n));
    for (i = 0; i < n; i++) for (j = i; j < n; j++) {
      var s = 0; for (d = 0; d < D; d++) s += X[i][d] * X[j][d];
      G[i][j] = s; G[j][i] = s;
    }
    for (i = 0; i < n; i++) trace += G[i][i];
    var eig = jacobiEigen(G, n).slice(0, nComp);
    var coords = [];
    for (i = 0; i < n; i++) coords.push([]);
    eig.forEach(function (e) {
      var sc = Math.sqrt(e.value);
      // deterministic sign: the anchor (first query) sits on the positive side
      var sign = e.vector[anchor || 0] < 0 ? -1 : 1;
      for (i = 0; i < n; i++) coords[i].push(sign * sc * e.vector[i]);
    });
    while (eig.length < nComp) { eig.push({ value: 0 }); for (i = 0; i < n; i++) coords[i].push(0); }
    return { coords: coords, ratio: eig.map(function (e) { return trace > 0 ? e.value / trace : 0; }) };
  }

  // ---------- vocabulary helpers ----------
  function normalize(w) { return (w || '').trim().toLowerCase().replace(/\s+/g, ' '); }
  function lookup(w) {
    w = normalize(w);
    if (!w) return null;
    if (index.has(w)) return w;
    var u = w.replace(/ /g, '_');
    if (index.has(u)) return u;
    return null;
  }

  function editDistance(a, b, max) {
    var la = a.length, lb = b.length;
    if (Math.abs(la - lb) > max) return max + 1;
    var prev = new Array(lb + 1), cur = new Array(lb + 1), i, j;
    for (j = 0; j <= lb; j++) prev[j] = j;
    for (i = 1; i <= la; i++) {
      cur[0] = i; var rowMin = i;
      for (j = 1; j <= lb; j++) {
        var cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
        if (cur[j] < rowMin) rowMin = cur[j];
      }
      if (rowMin > max) return max + 1;
      var t = prev; prev = cur; cur = t;
    }
    return prev[lb];
  }

  function closeSpellings(w, n) {
    var max = w.length <= 4 ? 1 : 2, hits = [];
    for (var i = 0; i < N; i++) {
      var dd = editDistance(w, vocab[i], max);
      if (dd <= max) hits.push({ w: vocab[i], d: dd, i: i });
    }
    hits.sort(function (a, b) { return a.d - b.d || a.i - b.i; });
    var out = hits.slice(0, n).map(function (h) { return h.w; });
    if (out.length < n) { // pad with prefix matches
      for (var j = 0; j < N && out.length < n; j++) if (vocab[j].indexOf(w.slice(0, 3)) === 0 && out.indexOf(vocab[j]) < 0) out.push(vocab[j]);
    }
    return out;
  }

  function prefixMatches(p, n) {
    var out = [];
    for (var i = 0; i < N && out.length < n; i++) if (vocab[i].lastIndexOf(p, 0) === 0) out.push(i);
    return out;
  }

  // ---------- messages ----------
  function showMsg(el, text, isErr, suggestions, onPick) {
    el.textContent = '';
    el.classList.toggle('err', !!isErr);
    if (!text) return;
    el.appendChild(document.createTextNode(text));
    if (suggestions && suggestions.length) {
      el.appendChild(document.createTextNode(' Did you mean '));
      suggestions.forEach(function (s, k) {
        var b = document.createElement('button');
        b.type = 'button'; b.textContent = s;
        b.addEventListener('click', function () { onPick(s); });
        el.appendChild(b);
        el.appendChild(document.createTextNode(k < suggestions.length - 2 ? ', ' : k === suggestions.length - 2 ? ' or ' : '?'));
      });
    }
  }

  function oovMessage(el, raw, onPick) {
    var w = normalize(raw);
    var sug = closeSpellings(w.replace(/ /g, '_'), 4);
    showMsg(el, '“' + w + '” is not in the ' + N.toLocaleString() + '-word vocabulary.', true, sug, onPick);
  }

  // ---------- explorer actions ----------
  function explore(raw, opts) {
    opts = opts || {};
    var w = lookup(raw);
    var msg = $('msg');
    if (!normalize(raw)) { showMsg(msg, 'Type a word first.', true); return false; }
    if (!w) { oovMessage(msg, raw, function (s) { $('wordInput').value = s; explore(s, opts); }); return false; }
    showMsg(msg, '');
    if (opts.add) {
      var at = queries.indexOf(w);
      if (at >= 0) active = at;
      else {
        if (queries.length >= MAX_QUERIES) { showMsg(msg, 'You can compare up to ' + MAX_QUERIES + ' words — remove one first.', true); return false; }
        queries.push(w); active = queries.length - 1;
      }
    } else {
      queries = [w]; active = 0;
    }
    $('wordInput').value = w;
    update();
    return true;
  }

  function removeQuery(k) {
    if (queries.length <= 1) return;
    queries.splice(k, 1);
    if (active >= queries.length) active = queries.length - 1;
    update();
  }

  function update() {
    queries.forEach(neighbors);
    buildPoints();
    renderCompare();
    renderNeighbors();
    renderPlots();
    try { history.replaceState(null, '', '#q=' + queries.map(encodeURIComponent).join(',')); } catch (e) {}
  }

  // ---------- point set ----------
  function buildPoints() {
    var pts = [], seen = new Map();
    queries.forEach(function (q, k) {
      seen.set(q, pts.length);
      pts.push({ w: q, i: index.get(q), cluster: k, isQuery: true, s: 1 });
    });
    queries.forEach(function (q, k) {
      neighbors(q).forEach(function (nb) {
        if (seen.has(nb.w)) return;
        seen.set(nb.w, pts.length);
        pts.push({ w: nb.w, i: nb.i, cluster: k, isQuery: false, s: nb.s });
      });
    });
    var res = pca(pts.map(function (p) { return row(p.i); }), 3, 0);
    pts.forEach(function (p, j) { p.c = res.coords[j]; });
    lastPoints = { pts: pts, seen: seen, ratio: res.ratio };
  }

  // ---------- rendering: lists ----------
  function chip(label, color, onClick, removable, onRemove) {
    var b = document.createElement('button');
    b.type = 'button'; b.className = 'chip';
    if (color) { var sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = color; b.appendChild(sw); }
    b.appendChild(document.createTextNode(label));
    if (removable) {
      var x = document.createElement('span'); x.className = 'x'; x.textContent = '×'; x.setAttribute('aria-hidden', 'true');
      b.appendChild(x);
      b.setAttribute('aria-label', 'Remove ' + label + ' from comparison');
      b.addEventListener('click', onRemove);
    } else b.addEventListener('click', onClick);
    return b;
  }

  function renderCompare() {
    var wrap = $('compareWrap'), box = $('compareChips');
    box.textContent = '';
    wrap.hidden = queries.length < 2;
    queries.forEach(function (q, k) {
      box.appendChild(chip(q, seriesColor(k), null, true, function () { removeQuery(k); }));
    });
  }

  function nbItem(word, score, color, onClick, maxScore) {
    var li = document.createElement('li');
    var b = document.createElement('button');
    b.type = 'button'; b.className = 'nb';
    b.style.setProperty('--c', color);
    b.title = 'Explore “' + word + '”';
    var main = document.createElement('span'); main.className = 'nb-main';
    var wEl = document.createElement('span'); wEl.className = 'nb-word'; wEl.textContent = word;
    var bar = document.createElement('span'); bar.className = 'nb-bar';
    var fill = document.createElement('span');
    fill.style.width = (Math.max(0, Math.min(1, score / (maxScore || 1))) * 100).toFixed(1) + '%';
    bar.appendChild(fill); main.appendChild(wEl); main.appendChild(bar);
    var sc = document.createElement('span'); sc.className = 'nb-score'; sc.textContent = score.toFixed(3);
    b.appendChild(main); b.appendChild(sc);
    b.addEventListener('click', onClick);
    li.appendChild(b);
    return li;
  }

  function renderNeighbors() {
    var tabs = $('nbTabs'), list = $('neighbors');
    tabs.textContent = ''; list.textContent = '';
    tabs.hidden = queries.length < 2;
    queries.forEach(function (q, k) {
      var t = document.createElement('button');
      t.type = 'button'; t.className = 'tab'; t.setAttribute('role', 'tab');
      t.setAttribute('aria-selected', String(k === active));
      t.style.setProperty('--c', seriesColor(k));
      var sw = document.createElement('span'); sw.className = 'sw';
      t.appendChild(sw); t.appendChild(document.createTextNode(q));
      t.addEventListener('click', function () { active = k; renderNeighbors(); });
      tabs.appendChild(t);
    });
    var q = queries[active];
    if (!q) return;
    var col = seriesColor(active);
    neighbors(q).forEach(function (nb) {
      list.appendChild(nbItem(nb.w, nb.s, col, function () { explore(nb.w); scrollToTop(); }, 1));
    });
  }

  function scrollToTop() {
    if (window.innerWidth <= 1000) $('explorer').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ---------- rendering: plots ----------
  var isTouch = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;

  function baseFont() { return { family: 'Inter, system-ui, sans-serif', color: cssVar('--muted'), size: 12 }; }
  function axisTitle(k) { return 'PC' + (k + 1) + ' · ' + (lastPoints.ratio[k] * 100).toFixed(1) + '%'; }

  function clusterTraces(is3d) {
    var pts = lastPoints.pts, traces = [];
    var text = cssVar('--text'), bg = cssVar('--card');
    var ax = function (p, k) { return p.c[k]; };
    queries.forEach(function (q, k) {
      var color = seriesColor(k);
      var qp = pts[lastPoints.seen.get(q)];
      // lines query → each of its neighbours
      var lx = [], ly = [], lz = [];
      neighbors(q).forEach(function (nb) {
        var p = pts[lastPoints.seen.get(nb.w)];
        lx.push(ax(qp, 0), ax(p, 0), null); ly.push(ax(qp, 1), ax(p, 1), null); lz.push(ax(qp, 2), ax(p, 2), null);
      });
      var line = { x: lx, y: ly, mode: 'lines', line: { color: color, width: is3d ? 3 : 1.4 }, opacity: is3d ? 0.55 : 0.45,
        hoverinfo: 'skip', showlegend: false, legendgroup: 'g' + k, type: is3d ? 'scatter3d' : 'scatter' };
      if (is3d) line.z = lz;
      traces.push(line);

      var own = pts.filter(function (p) { return p.cluster === k && !p.isQuery; });
      var nbT = {
        x: own.map(function (p) { return ax(p, 0); }), y: own.map(function (p) { return ax(p, 1); }),
        text: own.map(function (p) { return p.w; }),
        customdata: own.map(function (p) { return [p.w, p.s.toFixed(3), q]; }),
        mode: 'markers+text', type: is3d ? 'scatter3d' : 'scatter',
        textposition: own.map(function (p) { return ax(p, 1) < ax(qp, 1) ? 'bottom center' : 'top center'; }),
        textfont: { family: 'Inter, sans-serif', size: is3d ? 11 : 12, color: text },
        marker: { color: color, size: is3d ? 5 : 9, opacity: 0.95, line: { color: bg, width: is3d ? 0 : 1 } },
        hovertemplate: '<b>%{customdata[0]}</b><br>cos to “%{customdata[2]}”: %{customdata[1]}<extra></extra>',
        name: q, legendgroup: 'g' + k, showlegend: false
      };
      if (is3d) { nbT.z = own.map(function (p) { return ax(p, 2); }); nbT.textposition = 'top center'; }
      traces.push(nbT);

      var qT = {
        x: [ax(qp, 0)], y: [ax(qp, 1)], text: [q], customdata: [[q, '1.000', q]],
        mode: 'markers+text', type: is3d ? 'scatter3d' : 'scatter', textposition: 'top center',
        textfont: { family: 'Inter, sans-serif', size: is3d ? 14 : 15, color: color },
        marker: { color: color, size: is3d ? 10 : 18, symbol: is3d ? 'diamond' : 'circle', line: { color: is3d ? color : text, width: is3d ? 0 : 2.5 } },
        hovertemplate: '<b>%{customdata[0]}</b> (query)<extra></extra>',
        name: q, legendgroup: 'g' + k, showlegend: queries.length > 1
      };
      if (is3d) qT.z = [ax(qp, 2)];
      traces.push(qT);
    });
    return traces;
  }

  function renderPlots() {
    if (!window.Plotly || !lastPoints) return;
    var grid = cssVar('--grid'), zero = cssVar('--zero'), font = baseFont();
    var legend = { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 12, color: cssVar('--text') }, bgcolor: 'rgba(0,0,0,0)' };
    var axis2 = function (k) {
      return { title: { text: axisTitle(k), font: { size: 12 } }, gridcolor: grid, zerolinecolor: zero, linecolor: grid,
        tickfont: { size: 10 }, automargin: true };
    };
    var layout2 = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: font,
      margin: { l: 44, r: 16, t: queries.length > 1 ? 36 : 16, b: 40 },
      xaxis: axis2(0), yaxis: axis2(1), showlegend: queries.length > 1, legend: legend,
      hovermode: 'closest', dragmode: isTouch ? false : 'zoom',
      hoverlabel: { font: { family: 'Inter, sans-serif' } }
    };
    var axis3 = function (k) {
      return { title: { text: axisTitle(k), font: { size: 11 } }, gridcolor: grid, zerolinecolor: zero,
        showbackground: false, tickfont: { size: 9 }, showspikes: false };
    };
    var layout3 = {
      paper_bgcolor: 'rgba(0,0,0,0)', font: font, margin: { l: 0, r: 0, t: queries.length > 1 ? 30 : 0, b: 0 },
      showlegend: queries.length > 1, legend: legend,
      scene: { xaxis: axis3(0), yaxis: axis3(1), zaxis: axis3(2), aspectmode: 'cube',
        camera: { eye: { x: 1.45, y: 1.45, z: 0.9 } } },
      hoverlabel: { font: { family: 'Inter, sans-serif' } }
    };
    var cfg = { responsive: true, displaylogo: false, scrollZoom: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'],
      displayModeBar: isTouch ? false : 'hover' };

    Plotly.react('plot2d', clusterTraces(false), layout2, cfg);
    Plotly.react('plot3d', clusterTraces(true), layout3, Object.assign({}, cfg, { scrollZoom: true }));

    var r = lastPoints.ratio;
    $('var2d').textContent = 'variance shown: ' + ((r[0] + r[1]) * 100).toFixed(1) + '%';
    $('var3d').textContent = 'variance shown: ' + ((r[0] + r[1] + r[2]) * 100).toFixed(1) + '%';
  }

  function bindPlotClicks() {
    ['plot2d', 'plot3d'].forEach(function (id) {
      var el = $(id);
      el.on('plotly_click', function (e) {
        var p = e && e.points && e.points[0];
        if (!p || !p.customdata) return;
        var w = p.customdata[0];
        if (queries.indexOf(w) >= 0) { active = queries.indexOf(w); renderNeighbors(); return; }
        explore(w);
      });
    });
  }

  // ---------- autocomplete ----------
  var sugEl = $('suggestList'), input = $('wordInput'), sugItems = [], sugSel = -1;
  function closeSuggest() { sugEl.hidden = true; input.setAttribute('aria-expanded', 'false'); input.removeAttribute('aria-activedescendant'); sugSel = -1; }
  function openSuggest() {
    var p = normalize(input.value).replace(/ /g, '_');
    sugEl.textContent = ''; sugSel = -1;
    if (!p) { closeSuggest(); return; }
    sugItems = prefixMatches(p, 8);
    if (!sugItems.length || (sugItems.length === 1 && vocab[sugItems[0]] === p)) { closeSuggest(); return; }
    sugItems.forEach(function (i, k) {
      var li = document.createElement('li');
      li.id = 'sug-' + k; li.setAttribute('role', 'option'); li.setAttribute('aria-selected', 'false');
      var w = vocab[i];
      var lab = document.createElement('span');
      var m = document.createElement('mark'); m.textContent = w.slice(0, p.length);
      lab.appendChild(m); lab.appendChild(document.createTextNode(w.slice(p.length)));
      var rk = document.createElement('span'); rk.className = 'rank'; rk.textContent = '#' + (i + 1).toLocaleString();
      li.appendChild(lab); li.appendChild(rk);
      li.addEventListener('mousedown', function (e) { e.preventDefault(); input.value = w; closeSuggest(); explore(w); });
      sugEl.appendChild(li);
    });
    sugEl.hidden = false; input.setAttribute('aria-expanded', 'true');
  }
  function moveSel(dir) {
    var lis = sugEl.children;
    if (sugEl.hidden || !lis.length) return;
    if (sugSel >= 0) lis[sugSel].setAttribute('aria-selected', 'false');
    sugSel = (sugSel + 1 + dir + lis.length + 1) % (lis.length + 1) - 1;
    if (sugSel >= 0) {
      lis[sugSel].setAttribute('aria-selected', 'true');
      input.setAttribute('aria-activedescendant', lis[sugSel].id);
      lis[sugSel].scrollIntoView({ block: 'nearest' });
    } else input.removeAttribute('aria-activedescendant');
  }
  input.addEventListener('input', openSuggest);
  input.addEventListener('focus', function () { if (input.value) openSuggest(); });
  input.addEventListener('blur', function () { setTimeout(closeSuggest, 100); });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); moveSel(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); moveSel(-1); }
    else if (e.key === 'Escape') closeSuggest();
    else if (e.key === 'Enter' && sugSel >= 0) { e.preventDefault(); var w = vocab[sugItems[sugSel]]; input.value = w; closeSuggest(); explore(w); }
  });

  $('searchForm').addEventListener('submit', function (e) { e.preventDefault(); closeSuggest(); explore(input.value); });
  $('addBtn').addEventListener('click', function () { closeSuggest(); explore(input.value, { add: true }); });

  // ---------- analogy ----------
  function solveAnalogy() {
    var ids = ['anA', 'anB', 'anC'], words = [], msg = $('anMsg'), out = $('anResults');
    out.textContent = '';
    for (var k = 0; k < 3; k++) {
      var raw = $(ids[k]).value, w = lookup(raw);
      if (!normalize(raw)) { showMsg(msg, 'Fill in all three words.', true); return; }
      if (!w) {
        (function (field) { oovMessage(msg, raw, function (s) { $(field).value = s; solveAnalogy(); }); })(ids[k]);
        return;
      }
      $(ids[k]).value = w; words.push(w);
    }
    var a = row(index.get(words[0])), b = row(index.get(words[1])), c = row(index.get(words[2]));
    var v = new Float32Array(D), n = 0, d;
    for (d = 0; d < D; d++) { v[d] = b[d] - a[d] + c[d]; n += v[d] * v[d]; }
    n = Math.sqrt(n) || 1;
    for (d = 0; d < D; d++) v[d] /= n;
    var res = topK(v, 5, new Set(words.map(function (w) { return index.get(w); })));
    showMsg(msg, words[0] + ' : ' + words[1] + ' :: ' + words[2] + ' : ?', false);
    var col = cssVar('--s3');
    res.forEach(function (r) {
      out.appendChild(nbItem(r.w, r.s, col, function () { explore(r.w); $('explorer').scrollIntoView({ behavior: 'smooth' }); }, 1));
    });
  }
  $('analogyForm').addEventListener('submit', function (e) { e.preventDefault(); solveAnalogy(); });

  // ---------- model info ----------
  function renderModelInfo(bytes) {
    var dl = $('modelInfo');
    dl.textContent = '';
    function add(k, v, wide) {
      if (v === undefined || v === null || v === '') return;
      var div = document.createElement('div'); if (wide) div.className = 'wide';
      var dt = document.createElement('dt'); dt.textContent = k;
      var dd = document.createElement('dd'); dd.textContent = typeof v === 'object' ? JSON.stringify(v) : String(v);
      div.appendChild(dt); div.appendChild(dd); dl.appendChild(div);
    }
    add('Corpus', meta.corpus, true);
    add('Vocabulary', N.toLocaleString() + ' words');
    add('Dimensions', D);
    var params = meta.params || {};
    Object.keys(params).forEach(function (key) { add(key.replace(/_/g, ' '), params[key]); });
    add('Trained', meta.trained);
    add('Quantization', meta.quantization);
    add('Download', fmtMB(bytes));
    Array.prototype.forEach.call(document.querySelectorAll('.dim-n'), function (el) { el.textContent = D; });
  }

  // ---------- boot ----------
  function initialQueries() {
    var m = /[#&]q=([^&]*)/.exec(location.hash);
    if (!m) return [];
    return m[1].split(',').map(function (s) { try { return lookup(decodeURIComponent(s)); } catch (e) { return null; } })
      .filter(function (w, i, arr) { return w && arr.indexOf(w) === i; }).slice(0, MAX_QUERIES);
  }

  load().then(function () {
    $('loader').hidden = true;
    $('app').hidden = false;

    var ex = $('exampleChips');
    EXAMPLES.filter(function (w) { return index.has(w); }).forEach(function (w) {
      ex.appendChild(chip(w, null, function () { explore(w); }));
    });
    if (!ex.children.length) prefixMatches('', 6).forEach(function (i) { ex.appendChild(chip(vocab[i], null, function () { explore(vocab[i]); })); });

    var an = $('analogyChips');
    ANALOGIES.filter(function (t) { return t.every(function (w) { return index.has(w); }); }).forEach(function (t) {
      an.appendChild(chip(t[0] + ' : ' + t[1] + ' :: ' + t[2], null, function () {
        $('anA').value = t[0]; $('anB').value = t[1]; $('anC').value = t[2]; solveAnalogy();
      }));
    });

    renderModelInfo(N * D);
    bindPlotClicks();

    var init = initialQueries();
    if (init.length) { queries = init; active = 0; input.value = init[0]; update(); }
    else {
      var first = EXAMPLES.filter(function (w) { return index.has(w); })[0] || vocab[0];
      explore(first);
    }
    if (['anA', 'anB', 'anC'].every(function (id) { return lookup($(id).value); })) solveAnalogy();
  }).catch(fail);
})();
