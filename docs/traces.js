/* Agents vs. human experts: real trajectories from the TraceML paired split.
 * Renders into any element with [data-traces] from data/traces.json.
 * Add class "static" on that element to skip animation (used for image export). */
(function () {
  "use strict";
  var CLASS_LABEL = {
    submission: "tune the submission",
    model: "mutate the model",
    direction: "change or check direction",
    other: "other edits"
  };
  var CLASS_HINT = {
    submission: "re-weight, stack, post-process",
    model: "layers, epochs, seeds, knobs",
    direction: "swap model or features, change loss or CV, re-run to check",
    other: "fixes, refactors, infra"
  };
  var ORDER = ["submission", "model", "direction", "other"];

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function ordinal(n) {
    var s = ["th", "st", "nd", "rd"], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }
  var ICON = {
    agent: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="7" width="16" height="12" rx="3"/><circle class="f" cx="9.5" cy="13" r="1.4"/><circle class="f" cx="14.5" cy="13" r="1.4"/><path d="M12 7V4M9 19v2M15 19v2"/><circle class="f" cx="12" cy="3.2" r="1"/></svg>',
    human: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="7.5" r="3.5"/><path d="M4.5 20c.8-4 3.8-6.5 7.5-6.5s6.7 2.5 7.5 6.5"/></svg>'
  };

  // ---------------------------------------------------------------- tooltip
  var tip = null;
  function showTip(text, x, y) {
    if (!tip) { tip = el("div", "vs-tip"); tip.setAttribute("role", "status"); document.body.appendChild(tip); }
    tip.textContent = text;
    var w = Math.min(300, window.innerWidth - 24);
    tip.style.maxWidth = w + "px";
    tip.style.left = Math.max(12, Math.min(x + 14, window.innerWidth - w - 12)) + "px";
    tip.style.top = (y + 16) + "px";
    tip.style.opacity = "1";
  }
  function hideTip() { if (tip) tip.style.opacity = "0"; }
  function bindTip(node, text) {
    node.setAttribute("tabindex", "0");
    node.setAttribute("aria-label", text);
    node.addEventListener("mousemove", function (e) { showTip(text, e.clientX, e.clientY); });
    node.addEventListener("mouseleave", hideTip);
    node.addEventListener("focus", function () { var r = node.getBoundingClientRect(); showTip(text, r.left, r.bottom); });
    node.addEventListener("blur", hideTip);
  }

  // ---------------------------------------------------------------- one trajectory
  function sparkPaths(pct) {
    var n = pct.length, pts = [];
    for (var i = 0; i < n; i++) if (pct[i] !== null) pts.push([i / Math.max(n - 1, 1) * 1000, 100 - pct[i]]);
    if (!pts.length) return null;
    var d = "M" + pts[0][0].toFixed(1) + " " + pts[0][1].toFixed(1);
    for (var j = 1; j < pts.length; j++) d += "H" + pts[j][0].toFixed(1) + "V" + pts[j][1].toFixed(1);
    d += "H1000";
    return { line: d, area: d + "V100H" + pts[0][0].toFixed(1) + "Z", last: pts[pts.length - 1][1] };
  }

  function traceRow(t, meta) {
    var row = el("div", "trace trace-" + t.kind);
    var last = null;
    for (var k = t.pct.length - 1; k >= 0; k--) if (t.pct[k] !== null) { last = Math.round(t.pct[k]); break; }
    row.appendChild(el("div", "trace-head",
      '<span class="trace-name">' + t.label + '</span><span class="trace-meta">' + meta + "</span>" +
      '<span class="trace-final">' + (last === null ? "" : "ends at the <b>" + ordinal(last) + "</b> percentile") + "</span>"));

    var plot = el("div", "trace-plot");
    var p = sparkPaths(t.pct);
    plot.innerHTML =
      '<svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">' +
      '<line class="ref" x1="0" x2="1000" y1="10" y2="10"/><line class="ref" x1="0" x2="1000" y1="50" y2="50"/>' +
      (p ? '<path class="area" d="' + p.area + '"/><path class="line" d="' + p.line + '"/>' : "") + "</svg>" +
      '<span class="ref-label" style="top:10%">90th pct</span><span class="ref-label" style="top:50%">median</span>' +
      (p ? '<span class="end-dot" style="top:' + p.last + '%"></span>' : "") +
      '<span class="curtain" aria-hidden="true"></span>';
    plot.setAttribute("role", "img");
    plot.setAttribute("aria-label", t.label + ": best score so far, as a percentile of human trajectories" +
      (last === null ? "" : ", ending at the " + ordinal(last) + " percentile"));
    row.appendChild(plot);

    var rib = el("div", "ribbon" + (t.edits.length > 80 ? " dense" : ""));
    var total = 1400;
    t.edits.forEach(function (e, i) {
      var c = el("span", "cell c-" + e.c);
      c.style.setProperty("--d", Math.round(i / t.edits.length * total));
      bindTip(c, "Edit " + (i + 1) + " · " + CLASS_LABEL[e.c] + ": " + e.what.join(", ") + (e.why ? " (intent: " + e.why + ")" : ""));
      rib.appendChild(c);
    });
    row.appendChild(rib);
    return row;
  }

  function side(title, icon, traces, metas) {
    var s = el("div", "vs-side vs-" + icon);
    s.appendChild(el("div", "vs-side-head", '<span class="vs-icon">' + ICON[icon] + "</span><span>" + title + "</span>"));
    traces.forEach(function (t, i) { s.appendChild(traceRow(t, metas[i])); });
    return s;
  }

  // ---------------------------------------------------------------- difference plots
  function card(title, takeaway, body, source) {
    var c = el("div", "diff-card");
    c.appendChild(el("h3", "diff-title", title));
    c.appendChild(el("p", "diff-take", takeaway));
    c.appendChild(body);
    if (source) c.appendChild(el("p", "diff-src", source));
    return c;
  }

  function stackedShares(shares) {
    var wrap = el("div", "stacks");
    [["Codex", "Codex"], ["MLEvolve", "MLEvolve"], ["Top-10% humans", "Top 10% humans"], ["Other humans", "Other humans"]]
      .forEach(function (pair, r) {
        var sh = shares[pair[0]];
        if (!sh) return;
        var row = el("div", "stack-row");
        row.appendChild(el("span", "stack-name", pair[1]));
        var bar = el("span", "stack-bar");
        bar.style.setProperty("--r", r);
        ORDER.forEach(function (k) {
          var v = sh[k];
          if (!v) return;
          var seg = el("span", "seg c-" + k, v >= 0.14 ? Math.round(v * 100) + "%" : "");
          seg.style.width = (v * 100) + "%";
          bindTip(seg, pair[1] + ": " + Math.round(v * 100) + "% of edits " + CLASS_LABEL[k]);
          bar.appendChild(seg);
        });
        row.appendChild(bar);
        wrap.appendChild(row);
      });
    return wrap;
  }

  function simpleBars(items, max, refIndex) {
    var wrap = el("div", "sbars");
    items.forEach(function (it, i) {
      var row = el("div", "sbar-row");
      row.appendChild(el("span", "sbar-name", it.name));
      var track = el("span", "sbar-track");
      var fill = el("span", "sbar-fill" + (i === refIndex ? " ref" : ""));
      fill.style.width = Math.max(it.value / max * 100, it.value > 0 ? 1.2 : 0) + "%";
      fill.style.setProperty("--r", i);
      track.appendChild(fill);
      track.appendChild(el("span", "sbar-val", it.label));
      row.appendChild(track);
      bindTip(row, it.tip || (it.name + ": " + it.label));
      wrap.appendChild(row);
    });
    return wrap;
  }

  // ---------------------------------------------------------------- render
  function render(root, data) {
    var agents = data.traces.filter(function (t) { return t.kind !== "human"; });
    var humans = data.traces.filter(function (t) { return t.kind === "human"; });
    var grid = el("div", "vs-grid");
    grid.appendChild(side("AI agents", "agent", agents, ["64 edits in a 12-hour run", "8 edits along one search branch"]));
    grid.appendChild(el("div", "vs-divider", "<span>vs</span>"));
    grid.appendChild(side("Human experts", "human", humans, [humans[0].n_edits + " edits", humans[1].n_edits + " edits"]));
    root.appendChild(grid);

    var diffs = el("div", "diff-grid");
    diffs.appendChild(card("Where the edits go",
      "Codex spends 89% of its edits tuning the submission and MLEvolve 63% mutating its model. Humans spread theirs.",
      stackedShares(data.shares), "Every edit on the seven paired competitions."));
    var pv = data.paper.pivot;
    diffs.appendChild(card("How often they change direction",
      "Codex pivots too little and MLEvolve too much. Only the human pivots pay off.",
      simpleBars([
        { name: "Humans", value: pv.Humans, label: pv.Humans + "%", tip: "Humans pivot on 25% of transitions; the next three steps average +0.089." },
        { name: "Codex", value: pv.Codex, label: pv.Codex + "%", tip: "Codex pivots on 9% of transitions; matched to humans in the same state it is out-pivoted three to one." },
        { name: "MLEvolve", value: pv.MLEvolve, label: pv.MLEvolve + "%", tip: "MLEvolve pivots on 58% of transitions; the next three steps average −0.008." }
      ], 70, 0), "Share of transitions that pivot, from the paper."));
    var rt = data.paper.returns;
    diffs.appendChild(card("How often they go back",
      "Top humans reopen earlier approaches, and it usually pays. The agents almost never do.",
      simpleBars([
        { name: "Top humans", value: rt["Top humans"][0], label: "9%", tip: "Top humans return to earlier work on 9% of eligible versions; 78% of returns end higher." },
        { name: "Codex", value: rt.Codex[0], label: rt.Codex[1], tip: "Codex returns once in 658 eligible versions; the top-human rate predicts 60." },
        { name: "MLEvolve", value: rt.MLEvolve[0], label: rt.MLEvolve[1], tip: "MLEvolve never returns in 344 eligible versions; the top-human rate predicts 31." }
      ], 11, 0), "Share of eligible versions that return to earlier work, from the paper."));
    root.appendChild(diffs);

    if (root.classList.contains("static") || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      root.classList.add("played", "static");
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("played"); io.unobserve(en.target); }
      });
    }, { threshold: 0.2 });
    io.observe(grid);
    io.observe(diffs);
  }

  function legend(root) {
    var lg = el("div", "vs-legend");
    ORDER.forEach(function (k) {
      lg.appendChild(el("span", "lg-item", '<span class="sw c-' + k + '"></span><b>' + CLASS_LABEL[k] + "</b> " + CLASS_HINT[k]));
    });
    root.appendChild(lg);
  }

  document.querySelectorAll("[data-traces]").forEach(function (root) {
    fetch(root.getAttribute("data-traces"))
      .then(function (r) { return r.json(); })
      .then(function (data) { legend(root); render(root, data); root.setAttribute("data-ready", "1"); })
      .catch(function () { root.innerHTML = '<p class="vs-error">The interactive figure could not load.</p>'; });
  });
})();
