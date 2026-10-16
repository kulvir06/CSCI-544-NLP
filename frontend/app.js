const form = document.querySelector("#query-form");
const questionInput = document.querySelector("#question");
const dateInput = document.querySelector("#relevant-date");
const topKInput = document.querySelector("#top-k");
const candidateField = document.querySelector("#candidate-field");
const candidateInput = document.querySelector("#candidate-explanation");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const emptyState = document.querySelector("#empty-state");
const loadingState = document.querySelector("#loading-state");
const results = document.querySelector("#results");

let lastResult = null;

const labels = {
  answered: "Answer released",
  unsafe_blocked: "Unsafe draft blocked",
  abstained: "System abstained",
  no_evidence: "No applicable evidence",
};

const toneFor = (value) => {
  if (["answered", "safe", "pass", "preserved", "verified_applicable", "found"].includes(value)) return "safe";
  if (["unsafe_blocked", "unsafe", "fail", "missing", "contradicted", "verified_not_applicable"].includes(value)) return "unsafe";
  if (["abstained", "abstain", "review", "insufficient_temporal_evidence", "insufficient_evidence"].includes(value)) return "warning";
  return "neutral";
};

const humanize = (value) => (value || "not available").replaceAll("_", " ");

const safeSourceUrl = (value) => {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch (_error) {
    return null;
  }
};

function setMode() {
  const mode = new FormData(form).get("generator-mode");
  const provided = mode === "provided_candidate";
  candidateField.hidden = !provided;
  candidateInput.required = provided;
}

function showView(view) {
  emptyState.hidden = view !== "empty";
  loadingState.hidden = view !== "loading";
  results.hidden = view !== "results";
}

function badge(text, tone) {
  return `<span class="badge tone-${tone}">${text}</span>`;
}

function renderPipeline(data) {
  const verification = data.verification;
  const stages = [
    {
      number: "01 · Retrieval",
      label: data.retrieval.retrieval_status === "found" ? `${data.retrieval.hits.length} passage(s) found` : "No passage found",
      tone: toneFor(data.retrieval.retrieval_status),
    },
    {
      number: "02 · Temporal",
      label: humanize(data.retrieval.temporal_verdict),
      tone: toneFor(data.retrieval.temporal_verdict),
    },
    {
      number: "03 · Meaning",
      label: verification ? humanize(verification.meaning.verdict) : "Not run",
      tone: verification ? toneFor(verification.meaning.verdict) : "neutral",
    },
    {
      number: "04 · Release gate",
      label: labels[data.status] || humanize(data.status),
      tone: toneFor(data.status),
    },
  ];
  document.querySelector("#pipeline").innerHTML = stages.map((stage) => `
    <div class="pipeline-step ${stage.tone}">
      <span>${stage.number}</span>
      <strong>${stage.label}</strong>
    </div>
  `).join("");
}

function renderEvidence(data) {
  const hits = data.retrieval.hits || [];
  document.querySelector("#hit-count").textContent = `${hits.length} hit${hits.length === 1 ? "" : "s"}`;
  const list = document.querySelector("#evidence-list");
  if (!hits.length) {
    list.innerHTML = `<p class="no-data">The automatic temporal index did not contain a sufficiently relevant claim. Build the temporal index or inspect claims with unknown dates.</p>`;
    return;
  }

  list.innerHTML = hits.map((hit, index) => {
    const passage = hit.passage;
    const page = passage.page_number ? `Page ${passage.page_number}` : "Page unknown";
    const sourceUrl = safeSourceUrl(passage.source_url);
    const source = sourceUrl
      ? `<a class="source-link" href="${escapeAttribute(sourceUrl)}" target="_blank" rel="noreferrer">Source ↗</a>`
      : "No source URL";
    return `
      <div class="evidence-item">
        <div class="evidence-meta">
          <span class="evidence-rank">${index + 1}</span>
          <strong>${escapeHtml(passage.title || "Untitled document")}</strong>
          <span>·</span><span>${escapeHtml(passage.issuer || "Unknown issuer")}</span>
          <span>·</span><span>${page}</span>
          <span>·</span><span>${escapeHtml(data.retrieval.ranking_method)} ${hit.score.toFixed(3)}</span>
          ${passage.temporal_role ? badge(humanize(passage.temporal_role), toneFor(hit.temporal_status)) : ""}
          ${passage.temporal_confidence != null ? `<span>· temporal confidence ${Math.round(passage.temporal_confidence * 100)}%</span>` : ""}
          ${passage.temporal_scan_status ? badge(`scan: ${humanize(passage.temporal_scan_status)}`, passage.temporal_scan_status === "completed" ? "safe" : "warning") : ""}
          ${passage.block_index != null ? `<span>· document position ${passage.block_index}</span>` : ""}
          ${badge(humanize(hit.temporal_status), toneFor(hit.temporal_status))}
          ${source}
        </div>
        <p class="evidence-text">${escapeHtml(passage.text)}</p>
      </div>`;
  }).join("");
}

function renderMeaning(data) {
  const card = document.querySelector("#meaning-card");
  const verification = data.verification;
  card.hidden = !verification;
  if (!verification) return;

  const meaning = verification.meaning;
  const percentage = Math.round(meaning.critical_fact_preservation * 100);
  const ring = document.querySelector("#preservation-score");
  ring.style.setProperty("--score", percentage);
  ring.innerHTML = `<span>${percentage}%</span>`;
  document.querySelector("#fact-summary").innerHTML = [
    badge(`${meaning.preserved_fact_count} preserved`, "safe"),
    badge(`${meaning.missing_fact_count} missing`, meaning.missing_fact_count ? "unsafe" : "neutral"),
    badge(`${meaning.contradicted_fact_count} contradicted`, meaning.contradicted_fact_count ? "unsafe" : "neutral"),
  ].join("");

  const body = document.querySelector("#facts-body");
  if (!meaning.checks.length) {
    body.innerHTML = `<tr><td colspan="4">No deterministic atomic facts were found; human review is required.</td></tr>`;
    return;
  }
  body.innerHTML = meaning.checks.map((check) => `
    <tr title="${escapeAttribute(check.explanation)}">
      <td><span class="fact-type">${humanize(check.fact.fact_type)}</span></td>
      <td>${escapeHtml(check.fact.source_text)}</td>
      <td>${badge(humanize(check.status), toneFor(check.status))}</td>
      <td>${Math.round(check.lexical_coverage * 100)}%</td>
    </tr>
  `).join("");
}

function renderAnswer(data) {
  const card = document.querySelector("#answer-card");
  const text = data.answer || data.draft_explanation;
  card.hidden = !text;
  if (!text) return;

  const released = Boolean(data.answer);
  document.querySelector("#answer-card-title").textContent = released ? "Released answer" : "Blocked draft (inspection only)";
  const state = document.querySelector("#answer-state");
  state.textContent = released ? "Released" : "Not released";
  state.className = `badge tone-${released ? "safe" : "unsafe"}`;
  document.querySelector("#answer-text").textContent = text;
}

function renderResult(data) {
  lastResult = data;
  const tone = toneFor(data.status);
  document.querySelector("#verdict-title").textContent = labels[data.status] || humanize(data.status);
  document.querySelector("#verdict-message").textContent = data.message;
  const icon = document.querySelector("#verdict-icon");
  icon.textContent = tone === "safe" ? "✓" : tone === "unsafe" ? "×" : "!";
  icon.className = `verdict-icon tone-${tone}`;
  const verdictBadge = document.querySelector("#verdict-badge");
  verdictBadge.textContent = humanize(data.status);
  verdictBadge.className = `badge tone-${tone}`;

  renderPipeline(data);
  renderAnswer(data);
  renderEvidence(data);
  renderMeaning(data);
  document.querySelector("#raw-json").textContent = JSON.stringify(data, null, 2);
  showView("results");
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value);
  return node.innerHTML;
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

form.addEventListener("change", (event) => {
  if (event.target.name === "generator-mode") setMode();
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});

document.querySelector("#reset-button").addEventListener("click", () => {
  form.reset();
  setMode();
  formError.hidden = true;
  lastResult = null;
  showView("empty");
});

document.querySelector("#copy-json").addEventListener("click", async (event) => {
  if (!lastResult) return;
  await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
  const button = event.currentTarget;
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = "Copy JSON"; }, 1200);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.hidden = true;
  const mode = new FormData(form).get("generator-mode");
  const payload = {
    question: questionInput.value.trim(),
    relevant_date: dateInput.value || null,
    top_k: Number(topKInput.value),
    generator_mode: mode,
    candidate_explanation: mode === "provided_candidate" ? candidateInput.value.trim() : null,
  };

  submitButton.disabled = true;
  showView("loading");
  try {
    const response = await fetch("/prototype/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg).join("; ")
        : data.detail || "The request could not be completed.";
      throw new Error(detail);
    }
    renderResult(data);
  } catch (error) {
    showView(lastResult ? "results" : "empty");
    formError.textContent = error.message || "Could not connect to the local API.";
    formError.hidden = false;
  } finally {
    submitButton.disabled = false;
  }
});

setMode();
