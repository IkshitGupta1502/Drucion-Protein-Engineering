import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as $3Dmol from '3dmol';
import { api, type ExperimentResults } from '../api';

/**
 * Wild-type structure with the search painted onto it, scrubbable by round.
 *
 * What is real here, and what is not:
 *
 *  - The fold is the published AlphaFold2 model of the **wild-type** protein,
 *    served verbatim by the API. Residue numbering is verified server-side to
 *    match the assay sequence exactly, so position N really is residue N.
 *  - The colours are **real measured DMS values** for variants the optimizer
 *    paid to measure, revealed round by round.
 *  - The fold never changes as you scrub, and that is deliberate. No mutant
 *    structure exists in the dataset, and a single substitution does not
 *    visibly move a backbone. Animating a morph would be invented geometry.
 *    What moves is *where the optimizer has looked*.
 */

interface Props {
  results: ExperimentResults;
}

type Probe = { position: number; best: number; count: number; mutants: string[] };

/** Diverging red→grey→green ramp, normalised against the assay's own range. */
function fitnessColor(value: number, lo: number, mid: number, hi: number): string {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  if (value >= mid) {
    const t = clamp((value - mid) / Math.max(hi - mid, 1e-9));
    // grey → green
    return lerpHex([0.60, 0.63, 0.61], [0.07, 0.72, 0.42], t);
  }
  const t = clamp((mid - value) / Math.max(mid - lo, 1e-9));
  // grey → red
  return lerpHex([0.60, 0.63, 0.61], [0.85, 0.22, 0.22], t);
}

function lerpHex(a: number[], b: number[], t: number): string {
  const c = a.map((v, i) => Math.round(255 * (v + (b[i] - v) * t)));
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

export function Structure3D({ results }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ReturnType<typeof $3Dmol.createViewer> | null>(null);
  const [pdb, setPdb] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const rounds = useMemo(
    () => (results.rounds ?? []).filter((r) => r.selected.length > 0),
    [results.rounds],
  );
  const maxRound = rounds.length;
  const [round, setRound] = useState(maxRound);
  const [playing, setPlaying] = useState(false);

  // Follow the run forward as new rounds stream in, unless the user has
  // scrubbed back to look at something.
  const wasAtEnd = useRef(true);
  useEffect(() => {
    if (wasAtEnd.current) setRound(maxRound);
  }, [maxRound]);
  useEffect(() => {
    wasAtEnd.current = round >= maxRound;
  }, [round, maxRound]);

  // ---- data ---------------------------------------------------------

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPdb(null);
    fetch(api.structureUrl(results.assay_id))
      .then(async (r) => {
        if (!r.ok) {
          const detail = await r.json().catch(() => null);
          throw new Error(detail?.detail ?? `No structure available (${r.status})`);
        }
        return r.text();
      })
      .then((text) => {
        if (!cancelled) setPdb(text);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [results.assay_id]);

  /** Best measured value per residue, up to and including `round`. */
  const probes = useMemo(() => {
    const map = new Map<number, Probe>();
    rounds.slice(0, round).forEach((r) => {
      r.selected.forEach((v) => {
        v.positions.forEach((p) => {
          const cur = map.get(p);
          if (!cur) {
            map.set(p, { position: p, best: v.measured_fitness, count: 1, mutants: [v.mutant] });
          } else {
            cur.count += 1;
            cur.mutants.push(v.mutant);
            cur.best = Math.max(cur.best, v.measured_fitness);
          }
        });
      });
    });
    return map;
  }, [rounds, round]);

  /** Best variant discovered within the rounds currently shown. */
  const bestSoFar = useMemo(() => {
    let best: { mutant: string; fitness: number; positions: number[] } | null = null;
    rounds.slice(0, round).forEach((r) =>
      r.selected.forEach((v) => {
        if (!best || v.measured_fitness > best.fitness) {
          best = { mutant: v.mutant, fitness: v.measured_fitness, positions: v.positions };
        }
      }),
    );
    return best as { mutant: string; fitness: number; positions: number[] } | null;
  }, [rounds, round]);

  const scale = useMemo(() => {
    const stats = results.pool_stats;
    const values = [...probes.values()].map((p) => p.best);
    const lo = stats?.pool_min ?? (values.length ? Math.min(...values) : -1);
    const hi = stats?.pool_max ?? (values.length ? Math.max(...values) : 1);
    const mid = stats?.pool_mean ?? (lo + hi) / 2;
    return { lo, mid, hi };
  }, [results.pool_stats, probes]);

  // ---- viewer -------------------------------------------------------

  useEffect(() => {
    if (!pdb || !containerRef.current) return;
    const viewer = $3Dmol.createViewer(containerRef.current, {
      backgroundAlpha: 0,
      antialias: true,
    });
    viewer.addModel(pdb, 'pdb');
    viewer.setStyle({}, { cartoon: { color: '#c9d2cb', opacity: 1 } });
    viewer.zoomTo();
    viewer.render();
    viewerRef.current = viewer;

    // The container is laid out by the time the fetch resolves, but a
    // re-layout (sidebar, window resize) leaves the canvas stale, so keep it
    // in step. The initial call also re-fits a model that was measured
    // against a container mid-layout.
    const resize = () => {
      try {
        viewer.resize();
        viewer.render();
      } catch {
        /* viewer torn down mid-resize */
      }
    };
    const observer = new ResizeObserver(resize);
    if (containerRef.current) observer.observe(containerRef.current);
    const initial = setTimeout(resize, 50);

    return () => {
      clearTimeout(initial);
      observer.disconnect();
      try {
        viewer.clear();
      } catch {
        /* viewer already torn down */
      }
      viewerRef.current = null;
    };
  }, [pdb]);

  const paint = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Neutral backbone, so measured positions are the only thing that reads.
    // A single residue tinted inside a cartoon ribbon is invisible at this
    // scale, so each probed position gets an explicit bead on its CA atom.
    // `addStyle` rather than `setStyle` keeps the cartoon underneath.
    viewer.setStyle({}, { cartoon: { color: '#c9d2cb' } });

    probes.forEach((probe) => {
      const color = fitnessColor(probe.best, scale.lo, scale.mid, scale.hi);
      viewer.addStyle(
        { resi: probe.position, atom: 'CA' },
        { sphere: { radius: 1.5, color } },
      );
    });

    if (bestSoFar) {
      bestSoFar.positions.forEach((p) => {
        viewer.addStyle(
          { resi: p, atom: 'CA' },
          { sphere: { radius: 2.6, color: '#0f9d63' } },
        );
        viewer.addStyle({ resi: p }, { stick: { radius: 0.25, color: '#0f9d63' } });
        viewer.addLabel(
          `${bestSoFar!.mutant}  ${bestSoFar!.fitness.toFixed(2)}`,
          {
            // Placed at the centroid of the selection passed as the third
            // argument; `position` would need explicit XYZ coordinates.
            backgroundColor: '#0f2f22',
            backgroundOpacity: 0.85,
            fontColor: '#eafff5',
            fontSize: 11,
            inFront: true,
          },
          { resi: p, atom: 'CA' },
        );
      });
    }
    viewer.render();
  }, [probes, bestSoFar, scale]);

  // `pdb` is a dependency because the viewer is created in the effect above,
  // on the same commit where `pdb` first becomes non-null. On that commit
  // `paint` has not changed identity, so without `pdb` here this effect would
  // not re-run and the first paint would be silently skipped — leaving a bare
  // grey backbone with no measured residues on it.
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.removeAllLabels();
    paint();
  }, [paint, pdb]);

  // ---- playback -----------------------------------------------------

  useEffect(() => {
    if (!playing) return;
    if (round >= maxRound) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setRound((r) => Math.min(r + 1, maxRound)), 650);
    return () => clearTimeout(t);
  }, [playing, round, maxRound]);

  // ---- render -------------------------------------------------------

  if (error) {
    return (
      <div className="card">
        <div className="card-title">Protein structure</div>
        <div className="empty">
          No structure available for this protein. {error}
          <div className="note" style={{ marginTop: 8 }}>
            Structures ship with ProteinGym for most targets; run{' '}
            <code>python scripts/download_data.py</code> to fetch them.
          </div>
        </div>
      </div>
    );
  }

  const shown = rounds[Math.max(0, round - 1)];
  const measuredCount = probes.size;

  return (
    <div className="card structure-card">
      <div className="card-head">
        <div className="card-title">Where the search looked, in 3D</div>
        <div className="structure-legend">
          <span className="legend-key">
            <span className="swatch" style={{ background: '#d93838' }} /> low
          </span>
          <span className="legend-key">
            <span className="swatch" style={{ background: '#99a09c' }} /> neutral
          </span>
          <span className="legend-key">
            <span className="swatch" style={{ background: '#12b981' }} /> high
          </span>
        </div>
      </div>

      <div className="structure-stage">
        <div ref={containerRef} className="structure-viewer" />
        {loading && (
          <div className="structure-overlay">
            <span className="spinner" /> Loading structure…
          </div>
        )}
        <div className="structure-readout">
          <div>
            <strong>{measuredCount}</strong> residues probed
          </div>
          {bestSoFar && (
            <div>
              best so far <code>{bestSoFar.mutant}</code>{' '}
              <strong style={{ color: '#12b981' }}>
                {bestSoFar.fitness.toFixed(2)}
              </strong>
            </div>
          )}
        </div>
      </div>

      <div className="structure-controls">
        <button
          className="btn"
          onClick={() => {
            if (round >= maxRound) setRound(0);
            setPlaying((p) => !p);
          }}
          disabled={maxRound === 0}
        >
          {playing ? '❚❚ Pause' : '▶ Replay search'}
        </button>
        <input
          type="range"
          min={0}
          max={maxRound}
          value={round}
          onChange={(e) => {
            setPlaying(false);
            setRound(Number(e.target.value));
          }}
          aria-label="Optimization round"
          disabled={maxRound === 0}
        />
        <div className="structure-round">
          {round === 0 ? (
            <>before any measurement</>
          ) : (
            <>
              round <strong>{shown?.round ?? round}</strong> ·{' '}
              {shown?.n_measured ?? 0} measurements
            </>
          )}
        </div>
      </div>

      <div className="note">
        The fold is the published AlphaFold2 model of the <em>wild-type</em>
        protein; residue numbering is verified against the assay sequence
        server-side. Colours are real measured DMS values for variants the
        optimizer chose to measure. The backbone deliberately does not morph as
        you scrub — no mutant structure exists in the dataset, and a single
        substitution does not visibly move a backbone, so animating one would
        be invented geometry. What changes is where the search has looked.
      </div>
    </div>
  );
}
