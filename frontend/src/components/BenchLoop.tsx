import { useRef, useState } from 'react';
import {
  api,
  type ExperimentState,
  type IntakeReport,
  type ProposedBatch,
} from '../api';

/**
 * The prospective loop: propose a batch, take it to the bench, bring the
 * results back.
 *
 * The retrospective benchmark closes its loop against ProteinGym. This closes
 * it against a real experiment. Values that arrive here are tagged
 * `uploaded`, kept out of the published-data benchmark, and shown as such —
 * Drucion cannot audit a number somebody types into a spreadsheet, so the
 * honest thing is to label where it came from.
 */

interface Props {
  experiment: ExperimentState;
  /** Optional: the server also emits `round` and `status` over SSE when an
   *  upload is ingested, so the surrounding view refreshes on its own. */
  onIngested?: () => void;
}

export function BenchLoop({ experiment, onIngested }: Props) {
  const [batch, setBatch] = useState<ProposedBatch | null>(null);
  const [report, setReport] = useState<IntakeReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const ready = experiment.status !== 'preparing' && experiment.status !== 'created';

  const propose = async () => {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      setBatch(await api.propose(experiment.id));
    } catch (e) {
      // The planned round budget exists to keep benchmark comparisons fair;
      // a real campaign has no such stopping rule. Extend explicitly, tell
      // the user it happened, and carry on.
      const message = (e as Error).message;
      if (/already finished/i.test(message)) {
        try {
          const extended = await api.extend(experiment.id, 1);
          setBatch(await api.propose(experiment.id));
          setNotice(
            `Your planned rounds were already used, so this run was extended to ${extended.n_rounds}.`,
          );
        } catch (inner) {
          setError((inner as Error).message);
        }
      } else {
        setError(message);
      }
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.submitMeasurements(experiment.id, file);
      setReport(result);
      if (result.ingested) {
        setBatch(null);
        onIngested?.();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  return (
    <div className="card bench-loop">
      <div className="card-head">
        <div className="card-title">Run this batch in your own lab</div>
        <span className="badge">optional</span>
      </div>

      <p className="note" style={{ marginTop: 0 }}>
        Drucion picks the next batch, you measure it however you normally would,
        and you upload the results so it can learn and pick the next ten. Nothing
        here touches the published benchmark — your numbers are recorded as
        yours.
      </p>

      {error && <div className="alert err" role="alert">{error}</div>}
      {notice && <div className="alert info">{notice}</div>}

      <ol className="bench-steps">
        <li>
          <div className="bench-step-head">
            <span className="bench-step-num">1</span>
            <strong>Get the batch Drucion wants tested</strong>
          </div>
          <div className="btn-row">
            <button className="btn" onClick={propose} disabled={busy || !ready}>
              {batch ? `Round ${batch.round} ready` : 'Choose next batch'}
            </button>
            {batch && (
              <>
                <a
                  className="btn btn-primary"
                  href={api.worklistUrl(experiment.id)}
                  download
                >
                  ↓ Download worklist
                </a>
                <a
                  className="btn"
                  href={api.worklistUrl(experiment.id, true)}
                  download
                  title="Same sheet, pre-filled with the real published measurements for these variants, so you can test the upload path without inventing numbers"
                >
                  ↓ Pre-filled (for testing)
                </a>
              </>
            )}
          </div>
          {batch && (
            <div className="table-wrap scroll-y" style={{ marginTop: 12 }}>
              <table>
                <thead>
                  <tr>
                    <th>Variant</th>
                    <th className="num">Predicted</th>
                    <th className="num">Uncertainty</th>
                    <th className="num">To measure</th>
                  </tr>
                </thead>
                <tbody>
                  {batch.variants.map((v) => (
                    <tr key={v.index}>
                      <td className="mono">{v.mutant}</td>
                      <td className="num predicted">
                        {v.predicted_fitness != null
                          ? v.predicted_fitness.toFixed(3)
                          : '—'}
                      </td>
                      <td className="num predicted">
                        {v.predicted_uncertainty != null
                          ? `± ${v.predicted_uncertainty.toFixed(3)}`
                          : '—'}
                      </td>
                      <td className="num" style={{ color: 'var(--text-faint)' }}>
                        blank
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </li>

        <li>
          <div className="bench-step-head">
            <span className="bench-step-num">2</span>
            <strong>Measure them</strong>
          </div>
          <p className="note" style={{ margin: 0 }}>
            Fill in the <code>measured_fitness</code> column. Higher means
            better. Leave a row blank if it didn’t work — that variant goes back
            into the pool instead of being written off.
          </p>
        </li>

        <li>
          <div className="bench-step-head">
            <span className="bench-step-num">3</span>
            <strong>Upload the results</strong>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            disabled={busy || !batch}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) upload(f);
            }}
          />
          {!batch && (
            <div className="note">Choose a batch first, so there is something to match the sheet against.</div>
          )}
        </li>
      </ol>

      {report && <IntakeSummary report={report} />}
    </div>
  );
}

function IntakeSummary({ report }: { report: IntakeReport }) {
  const problems = report.n_rejected > 0 || report.n_missing > 0;
  return (
    <div className={`alert ${report.ingested ? (problems ? 'warn' : 'info') : 'err'}`}>
      <strong>
        {report.ingested
          ? `Round ${report.round}: ${report.n_accepted} measurement${
              report.n_accepted === 1 ? '' : 's'
            } added`
          : 'Nothing was added'}
      </strong>
      <ul style={{ margin: '8px 0 0', paddingLeft: 20 }}>
        {report.n_skipped > 0 && (
          <li>
            {report.n_skipped} row{report.n_skipped === 1 ? '' : 's'} left blank —
            returned to the pool: <code>{report.skipped.join(', ')}</code>
          </li>
        )}
        {report.n_missing > 0 && (
          <li>
            {report.n_missing} proposed variant
            {report.n_missing === 1 ? '' : 's'} not in the sheet:{' '}
            <code>{report.missing.join(', ')}</code>
          </li>
        )}
        {report.rejected.map((r) => (
          <li key={r.row}>
            <code>{r.row}</code> rejected — {r.reason}
          </li>
        ))}
      </ul>
      {report.ingested && (
        <div style={{ marginTop: 8 }}>
          The model has been retrained on your results. Choose the next batch to
          continue.
        </div>
      )}
    </div>
  );
}
