const spinner = document.getElementById("spinner");
const output = document.getElementById("output");
const agentsEl = document.getElementById("agents");
const taskEl = document.getElementById("task");
const runBtn = document.getElementById("runBtn");

let poller = null;

// 🔹 GLOBAL state (FIX #1)
let expectedAgents = [];
let finishedAgents = new Set();

async function run() {
  const task = taskEl.value.trim();
  if (!task) return;

  runBtn.disabled = true;
  runBtn.style.display = "none";
  spinner.style.display = "block";
  agentsEl.innerHTML = "";
  output.innerHTML = "";
  finishedAgents.clear();

  const res = await fetch("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task })
  });

  const data = await res.json();

  // ---- render agents (initial, spinning)
  if (data.agents) {
    expectedAgents = data.agents.slice(); // keep original names with emoji

    agentsEl.innerHTML = `
      <div class="agents-list">
        ${expectedAgents.map(a => agentBox(a, true)).join("")}
      </div>
    `;
    spinner.style.display = "none";
  }

  if (poller) clearInterval(poller);
  poller = setInterval(fetchResults, 5000);
  fetchResults();
}

async function fetchResults() {
  const res = await fetch("/results");
  const data = await res.json();

  if (!data || Object.keys(data).length === 0) return;

  renderResults(data);
}

function renderResults(results) {
  output.innerHTML = "";

  const agents = Array.isArray(results)
    ? results
    : Object.entries(results).map(([agent, reports]) => ({ agent, reports }));

  agents.forEach(a => finishedAgents.add(cleanName(a.agent)));

  // ---- update agent icons (FIX #2)
  const allFinished = finishedAgents.size === expectedAgents.length;

  // Map agent labels to their results for quick lookup
  const agentResultsMap = {};
  agents.forEach(agentObj => {
    let label = agentObj.agent.replace(/Agent\b.*$/,"").replace(/\p{Extended_Pictographic}/gu,"").trim().split(" ")[0];
    agentResultsMap[label] = agentObj;
  });

  // Render agent icons with click handlers
  agentsEl.innerHTML = `
    <div class="agents-list">
      ${expectedAgents.map(name => {
        let label = name.replace(/Agent\b.*$/,"").replace(/\p{Extended_Pictographic}/gu,"").trim().split(" ")[0];
        let spinning = !finishedAgents.has(label);
        // Add data-label for click handler
        return `
          <div class="agent-box-wrapper" data-label="${label}" style="display:inline-block;cursor:${spinning ? 'default' : 'pointer'};">
            ${agentBox(name, spinning)}
          </div>
        `;
      }).join("")}
    </div>
    ${
      allFinished
        ? `<div class="done">✅ All agents finished<br><span class="done-instruction">Click an agent icon to view its results.</span></div>`
        : ""
    }
  `;

  // Remove agent results from main output, just show instructions
  output.innerHTML = `<div class="agent-results-instructions">Click an agent icon to view its results.</div>`;

  // Add modal HTML if not present
  if (!document.getElementById("agentModal")) {
    const modalDiv = document.createElement("div");
    modalDiv.id = "agentModal";
    modalDiv.style.display = "none";
    modalDiv.innerHTML = `
      <div id="agentModalBackdrop" style="position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.3);z-index:1000;"></div>
      <div id="agentModalContent" style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#fff;padding:2em 1.5em;z-index:1001;max-width:90vw;max-height:80vh;overflow:auto;border-radius:10px;box-shadow:0 2px 16px #0002;">
        <button id="agentModalClose" style="position:absolute;top:0.5em;right:0.7em;font-size:1.3em;background:none;border:none;cursor:pointer;">&times;</button>
        <div id="agentModalBody"></div>
      </div>
    `;
    document.body.appendChild(modalDiv);
    // Close modal on click
    document.getElementById("agentModalClose").onclick = hideAgentModal;
    document.getElementById("agentModalBackdrop").onclick = hideAgentModal;
  }

  // Attach click handlers to agent icons
  document.querySelectorAll(".agent-box-wrapper").forEach(el => {
    const label = el.getAttribute("data-label");
    if (finishedAgents.has(label)) {
      el.onclick = () => showAgentModal(label, agentResultsMap[label]);
    } else {
      el.onclick = null;
    }
  });

  // ---- stop polling when done (FIX #3)
  if (finishedAgents.size === expectedAgents.length) {
    clearInterval(poller);
    spinner.style.display = "none";
    runBtn.disabled = false;
    runBtn.style.display = "";
  }
}

// Modal helpers
function showAgentModal(label, agentObj) {
  const modal = document.getElementById("agentModal");
  const body = document.getElementById("agentModalBody");
  if (!agentObj) {
    body.innerHTML = `<div>No results for this agent.</div>`;
  } else {
    body.innerHTML = `
      <h2 style="margin-top:0;">Directives for ${label}</h2>
      ${agentObj.reports.map(r => {
        let parsed = {};
        try { parsed = JSON.parse(r.results); } catch {}
        return `
          <div class="cap" style="margin-bottom:1em;padding-bottom:1em;border-bottom:1px solid #eee;">
            <p><span class="result-label">Capability applied:</span> ${r.capability}</p>
            ${parsed.recommendation ? `
              <p><span class="result-label">Recommendation:</span><br>${parsed.recommendation}</p>
            ` : ""}
            ${parsed.explanation ? `
              <p><span class="result-label">Explanation:</span><br>${parsed.explanation}</p>
            ` : ""}
            ${r.validation ? `
              <p><span class="result-label">Validation result:</span> ${r.validation}</p>
            ` : ""}
          </div>
        `;
      }).join("")}
    `;
  }
  modal.style.display = "block";
}

function hideAgentModal() {
  const modal = document.getElementById("agentModal");
  if (modal) modal.style.display = "none";
}

// ---------- helpers ----------
function cleanName(name) {
  return (name || "")
    .replace(/Agent.*$/,"")
    .replace(/[\u{1F300}-\u{1FAFF}\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu,"")
    .trim()
    .split(" ")[0];
}

function extractEmoji(str) {
  // Get the last "word" (could be emoji or text)
  let parts = str.trim().split(" ");
  let last = parts[parts.length - 1];
  // If last contains at least one emoji codepoint, treat as emoji
  if (/\p{Extended_Pictographic}/u.test(last)) {
    return last;
  }
  return "";
}

function agentBox(name, spinning) {
  let emoji = extractEmoji(name);
  if (!emoji) emoji = "🤖";
  let label = name.replace(/Agent.*$/,"").replace(/\p{Extended_Pictographic}/gu,"").trim().split(" ")[0];
  return `
    <div class="agent-box">
      <div class="agent-circle" style="position:relative;">
        ${spinning ? '<div class="agent-spinner"></div>' : ''}
        <span style="position:relative; z-index:2; font-size:1.7em;">${emoji}</span>
      </div>
      <div class="agent-label">${label}</div>
    </div>
  `;
}