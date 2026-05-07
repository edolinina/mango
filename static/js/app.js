const { useState, useEffect, useRef } = React;

function cleanName(name) {
  return (name || "")
    .replace(/Agent.*$/, "")
    .replace(/[\u{1F300}-\u{1FAFF}\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "")
    .trim()
    .split(" ")[0];
}

function agentKey(name) {
  return (name || "")
    .replace(/[\u{1F300}-\u{1FAFF}\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "")
    .trim()
    .split(" ")[0];
}

function extractEmoji(str) {
  const parts = str.trim().split(" ");
  const last = parts[parts.length - 1];
  return /\p{Extended_Pictographic}/u.test(last) ? last : "🤖";
}

function mlValidationBadgeClass(status) {
  if (status === "success") return "pass";
  if (status === "partial_pass") return "partial";
  return "fail";
}

function mlValidationLabel(status) {
  if (status === "success") return "✓ Success";
  if (status === "partial_pass") return "~ Partial Pass";
  return "✗ Fail";
}

function AgentCard({ name, spinning, onClick }) {
  const emoji = extractEmoji(name);
  const label = cleanName(name);
  return (
    <div className={`agent-card ${spinning ? "" : "ready"}`} onClick={!spinning ? onClick : undefined}>
      <div className="agent-avatar">
        <div className="emoji">{emoji}</div>
        {spinning && <div className="ring" />}
      </div>
      <div className="agent-info">
        <div className="name">{label}</div>
        <div className="agent-meta-row">
          <span className={`status-tag ${spinning ? "running" : "done"}`}>
            {spinning ? "Running" : "Complete"}
          </span>
        </div>
      </div>
    </div>
  );
}

function AgentModal({ label, agentObj, onClose }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{extractEmoji(agentObj?.agent || "")} {label} — Results</h2>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          {agentObj ? agentObj.reports.map((r, i) => {
            let parsed = {};
            try { parsed = JSON.parse(r.results); } catch(e) {}
            return (
              <div key={i} className="directive-block">
                <div className="cap-label">{r.capability}</div>
                {parsed.recommendation && (
                  <div className="field">
                    <div className="field-key">Recommendation</div>
                    <div className="field-val">{parsed.recommendation}</div>
                  </div>
                )}
                {parsed.explanation && (
                  <div className="field">
                    <div className="field-key">Explanation</div>
                    <div className="field-val">{parsed.explanation}</div>
                  </div>
                )}
                {parsed.next_steps && parsed.next_steps.length > 0 && (
                  <div className="field">
                    <div className="field-key">Next Steps</div>
                    <div className="field-val">
                      <ol style={{ paddingLeft: "1.2em", margin: 0 }}>
                        {parsed.next_steps.map((s, j) => <li key={j}>{s}</li>)}
                      </ol>
                    </div>
                  </div>
                )}
                {parsed.self_validation && (
                  <div className="field">
                    <div className="field-key">Self Validation</div>
                    <span className={`validation-badge ${parsed.self_validation.status === "pass" ? "pass" : "fail"}`}>
                      {parsed.self_validation.status === "pass" ? "✓ Pass" : "✗ Fail"}
                    </span>
                  </div>
                )}
                {parsed.iterations != null && (
                  <div className="field">
                    <div className="field-key">Iterations</div>
                    <div className="field-val">{parsed.iterations}</div>
                  </div>
                )}
                {parsed.ml_validation && (
                  <div className="field">
                    <div className="field-key">ML Validation</div>
                    {parsed.ml_validation.status === "error" ? (
                      <span className="validation-badge fail" title={parsed.ml_validation.error}>⚠ Unavailable</span>
                    ) : (
                      <div className="validation-ml-row">
                        <span className={`validation-badge ${mlValidationBadgeClass(parsed.ml_validation.status)}`}>
                          {mlValidationLabel(parsed.ml_validation.status)}
                        </span>
                        <span className="validation-ml-stats">
                          {parsed.ml_validation.pass_rate}% pass rate · {parsed.ml_validation.passed}/{parsed.ml_validation.total} samples
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          }) : <p style={{ color: "var(--text-muted)" }}>No results available.</p>}
        </div>
      </div>
    </div>
  );
}

function ApprovalModal({ directivesText, onApprove, onReject }) {
  return (
    <div className="modal-backdrop">
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Review &amp; Approve Task</h2>
        </div>
        <div className="approval-body">
          <p>The Central Executive has generated the following directives. Review and approve to continue, or reject to cancel.</p>
          {directivesText && (
            <div className="directives-preview">{directivesText}</div>
          )}
          <div className="approval-actions">
            <button className="btn btn-ghost" onClick={onReject}>Reject</button>
            <button className="btn btn-success" onClick={onApprove}>Approve &amp; Run</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [task, setTask] = useState("");
  const [phase, setPhase] = useState("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [pendingDirectives, setPendingDirectives] = useState(null);
  const [expectedAgents, setExpectedAgents] = useState([]);
  const [finishedAgents, setFinishedAgents] = useState(new Set());
  const [agentResultsMap, setAgentResultsMap] = useState({});
  const [selectedAgent, setSelectedAgent] = useState(null);
  const pollerRef = useRef(null);
  const expectedLabelsRef = useRef([]);

  const expectedLabels = expectedAgents.map((name) => agentKey(name));
  const allFinished = expectedLabels.length > 0 && expectedLabels.every((label) => finishedAgents.has(label));

  useEffect(() => {
    if (allFinished && pollerRef.current) {
      clearInterval(pollerRef.current);
      pollerRef.current = null;
      setPhase("done");
      setStatusMsg("");
    }
  }, [allFinished]);

  async function handleRun() {
    if (!task.trim()) return;
    setPhase("running");
    setStatusMsg("Generating directives…");
    setErrorMsg("");
    setExpectedAgents([]);
    expectedLabelsRef.current = [];
    setFinishedAgents(new Set());
    setAgentResultsMap({});

    const res = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
    });
    const data = await res.json();

    if (data.status === "error") {
      setPhase("idle");
      setStatusMsg("");
      setErrorMsg(data.error || "Failed to start workflow.");
      return;
    }

    if (data.status === "pending") {
      setPendingDirectives(data.directives);
      setPhase("approval");
      setStatusMsg("");
      return;
    }

    startPolling(data);
  }

  async function handleApprove() {
    setPhase("running");
    setStatusMsg("Dispatching to agents…");
    setErrorMsg("");
    setPendingDirectives(null);
    const res = await fetch("/approve", { method: "POST" });
    const data = await res.json();

    if (data.status === "error") {
      setPhase("idle");
      setStatusMsg("");
      setErrorMsg(data.error || "Failed to approve workflow.");
      return;
    }

    startPolling(data);
  }

  async function handleReject() {
    await fetch("/reject", { method: "POST" });
    setPhase("idle");
    setStatusMsg("");
    setErrorMsg("");
    setPendingDirectives(null);
    expectedLabelsRef.current = [];
  }

  function startPolling(data) {
    if (data.agents) {
      const nextAgents = data.agents.slice();
      setExpectedAgents(nextAgents);
      expectedLabelsRef.current = nextAgents.map((name) => agentKey(name));
    }
    setStatusMsg("Agents are working…");
    if (pollerRef.current) clearInterval(pollerRef.current);
    pollerRef.current = setInterval(fetchResults, 5000);
    fetchResults();
  }

  async function fetchResults() {
    const res = await fetch("/results");
    const data = await res.json();
    if (!data) return;

    if (!Array.isArray(data) && typeof data === "object" && data.status === "error") {
      if (pollerRef.current) {
        clearInterval(pollerRef.current);
        pollerRef.current = null;
      }
      setPhase("idle");
      setStatusMsg("");
      setErrorMsg(data.error || "Workflow failed.");
      return;
    }

    // Ignore other control/status payloads from the API.
    if (!Array.isArray(data) && typeof data === "object" && (data.status || data.error)) {
      return;
    }

    if (!Array.isArray(data) && Object.keys(data).length === 0) return;

    const agents = Array.isArray(data)
      ? data
      : Object.entries(data).map(([agent, reports]) => ({ agent, reports }));

    // Only count agents that are part of this run.
    const expectedSet = new Set(expectedLabelsRef.current);
    const relevantAgents = agents.filter((a) => expectedSet.has(agentKey(a.agent)));

    if (relevantAgents.length === 0) return;

    const newMap = {};
    relevantAgents.forEach(agentObj => { newMap[agentKey(agentObj.agent)] = agentObj; });

    setAgentResultsMap(prev => ({ ...prev, ...newMap }));
    setFinishedAgents(prev => {
      const next = new Set(prev);
      relevantAgents.forEach(a => next.add(agentKey(a.agent)));
      return next;
    });
  }

  const busy = phase === "running" || phase === "approval";

  return (
    <>
      <header className="app-topbar">
        <div className="app-topbar-inner">
          <div className="brand-lockup">
            <span className="logo">🥭</span>
            <div>
              <div className="eyebrow">Executive Workspace</div>
              <h1 className="app-topbar-title">MANGO Operator</h1>
              <div className="subtitle">Multi-agent orchestration for business decisions</div>
            </div>
          </div>
          <div className="app-nav-actions">
            <div className="count-pill">{expectedAgents.length || 0} agents</div>
          </div>
        </div>
      </header>

      <main className="app-shell">
        <section className="panel hero-panel">
          <div className="hero">
            <p className="eyebrow">Operational Brief</p>
            <h1>Set one clear business goal.</h1>
            <p className="hero-copy">
              Submit one decision objective, dispatch the executive plan, and review each agent's
              recommendation in a single workspace.
            </p>
          </div>
          <div className="hero-stats">
            <div>
              <span className="summary-label">Workflow state</span>
              <strong>{phase === "idle" ? "Ready" : phase === "approval" ? "Waiting for approval" : phase === "done" ? "Completed" : "In progress"}</strong>
            </div>
            <div>
              <span className="summary-label">Completed agents</span>
              <strong>{finishedAgents.size} / {expectedAgents.length || 0}</strong>
            </div>
          </div>
        </section>

        <div className="workspace-grid">
          <div className="workspace-column">
            <section className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Task Input</p>
                  <h2>Define the objective</h2>
                </div>
              </div>

              <label className="field">
                <span>Business objective</span>
                <textarea
                  value={task}
                  onChange={e => setTask(e.target.value)}
                  placeholder="Describe the business objective you want agents to work on…"
                  disabled={busy}
                />
              </label>

              <p className="field-help">
                Be specific about the outcome you want balanced across profitability, customers, and workforce.
              </p>

              <div className="action-strip">
                <div className="action-controls-inline">
                  <button className="btn primary-button run-button" onClick={handleRun} disabled={busy || !task.trim()}>
                    {phase === "running" ? "Running…" : "Run task"}
                  </button>
                </div>
              </div>

              {errorMsg && <p className="field-error">{errorMsg}</p>}

            </section>
          </div>

          <div className="workspace-column">
            <section className="panel results-panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Execution</p>
                  <h2>Agent progress</h2>
                </div>
                <div className="count-pill">{finishedAgents.size} complete</div>
              </div>

              <div className="results-summary">
                <div>
                  <span className="summary-label">Expected</span>
                  <strong>{expectedAgents.length || 0}</strong>
                </div>
                <div>
                  <span className="summary-label">Finished</span>
                  <strong>{finishedAgents.size}</strong>
                </div>
                <div>
                  <span className="summary-label">Status</span>
                  <strong>{allFinished ? "Ready to review" : expectedAgents.length ? "Running" : "Waiting"}</strong>
                </div>
              </div>

              {expectedAgents.length > 0 ? (
                <>
                  <div className="section-title">Agents</div>
                  <div className="agents-grid">
                    {expectedAgents.map(name => {
                      const key = agentKey(name);
                      const label = cleanName(name);
                      return (
                        <AgentCard
                          key={name}
                          name={name}
                          spinning={!finishedAgents.has(key)}
                          onClick={() => setSelectedAgent(key)}
                        />
                      );
                    })}
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  Agents will appear here after the Central Executive generates a plan.
                </div>
              )}

              {allFinished && (
                <div className="done-banner">
                  <span>✓</span>
                  <span>All agents completed. Select any card to inspect the detailed recommendation.</span>
                </div>
              )}
            </section>
          </div>
        </div>
      </main>

      {selectedAgent && (
        <AgentModal
          label={selectedAgent}
          agentObj={agentResultsMap[selectedAgent]}
          onClose={() => setSelectedAgent(null)}
        />
      )}

      {phase === "approval" && (
        <ApprovalModal
          directivesText={pendingDirectives}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      )}
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
