import type { ExperimentResults } from '../api';

export function BudgetEstimate({ estimate }: { estimate: ExperimentResults['budget_estimate'] }) {
  if (!estimate) return null;
  const saved = estimate.saved_measurements >= 1;
  const extra = estimate.saved_measurements <= -1;
  return <section className="card" aria-label="Estimated budget compared with random sampling">
    <div className="card-head"><div className="card-title">Your budget compared with random sampling</div><span className="badge">Retrospective estimate</span></div>
    <h2 style={{ fontSize: 26, color: saved ? 'var(--accent)' : 'var(--text)', marginBottom: 12 }}>
      {saved ? `About ${Math.round(estimate.saved_percent)}% fewer measurements` : extra ? 'No estimated saving at this score' : 'About the same measurement budget'}
    </h2>
    <p style={{ lineHeight: 1.8 }}>Your search first reached a score of <strong>{estimate.target_score.toFixed(3)}</strong> after <strong>{estimate.actual_measurements} measurements</strong>. Random sampling would need about <strong>{Math.round(estimate.random_expected_measurements)} on average</strong> to reach an equal or better score.
      {saved ? ` That’s about ${Math.round(estimate.saved_measurements)} measurements saved.` : extra ? ` Your run used about ${Math.round(-estimate.saved_measurements)} more measurements to reach this score.` : ''}
    </p>
    <details style={{ marginTop: 12 }}><summary>How this estimate works</summary><p className="note" style={{ marginTop: 8 }}>Calculated from published dataset scores using random sampling without replacement, with the same initial measurements, batch size, and excluded holdout. Includes the initial sample and allows random sampling to continue beyond this run’s budget. Compares the first round that reached your current best score; later measurements are not counted as savings. This is an average for this score, not a guaranteed saving or a general performance claim. No extra model training or experiments are run.</p></details>
  </section>;
}
