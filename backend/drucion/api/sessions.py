"""Server-side experiment sessions: state, threading, and event fan-out.

Scientific computation stays in :mod:`drucion.optimization`; this module only
owns lifecycle (create / step / run / pause), persistence and the event queues
that feed Server-Sent Events to the dashboard.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RESULTS_DIR, ExperimentConfig
from ..data.proteingym import load_assay
from ..embeddings.features import FeatureSet, build_features
from ..experiments.benchmark import run_benchmark, save_benchmark
from ..experiments.store import Store
from ..optimization.loop import ClosedLoopOptimizer

logger = logging.getLogger(__name__)

#: Features are expensive and label-free, so they are cached per
#: (assay, featurizer, components) across experiments in the process.
_FEATURE_CACHE: dict[tuple, FeatureSet] = {}
_FEATURE_LOCK = threading.Lock()


def get_features(config: ExperimentConfig, pool, progress=None) -> FeatureSet:
    key = (
        config.assay_id,
        config.pool_size,
        config.embedding.featurizer,
        config.embedding.n_components,
        config.embedding.model_name,
        config.embedding.include_zero_shot,
    )
    with _FEATURE_LOCK:
        cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    fs = build_features(pool, config.embedding, progress=progress)
    with _FEATURE_LOCK:
        _FEATURE_CACHE[key] = fs
    return fs


@dataclass
class Session:
    """One live experiment."""

    id: str
    config: ExperimentConfig
    status: str = "created"  # created|preparing|ready|running|paused|done|error
    optimizer: ClosedLoopOptimizer | None = None
    error: str | None = None
    subscribers: list[queue.Queue] = field(default_factory=list)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- events --------------------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    def emit(self, event: str, data: dict[str, Any]) -> None:
        payload = {"event": event, "data": data, "ts": time.time()}
        for q in list(self.subscribers):
            try:
                q.put_nowait(payload)
            except queue.Full:  # slow client: drop rather than block science
                pass


class SessionManager:
    """Owns every live experiment and benchmark job in the process."""

    def __init__(self, store: Store | None = None) -> None:
        self.store = store or Store()
        self.sessions: dict[str, Session] = {}
        self.benchmarks: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def create(self, config: ExperimentConfig) -> Session:
        exp_id = uuid.uuid4().hex[:12]
        session = Session(id=exp_id, config=config)
        with self._lock:
            self.sessions[exp_id] = session
        self.store.create_experiment(exp_id, config.to_dict(), config.total_budget)

        def prepare() -> None:
            try:
                session.status = "preparing"
                self.store.update_experiment(exp_id, status="preparing")
                session.emit("status", {"status": "preparing"})
                assay = load_assay(config.assay_id, pool_size=config.pool_size)

                def feat_progress(done: int, total: int) -> None:
                    session.emit(
                        "embedding_progress", {"done": done, "total": total}
                    )

                features = get_features(config, assay.pool, progress=feat_progress)

                # Give interactive runs the same fixed evaluation holdout the
                # benchmark uses, so the dashboard can show a genuine
                # out-of-sample surrogate-accuracy curve. The holdout is
                # excluded from the candidate pool and costs no budget.
                pool_size = len(assay.pool)
                holdout: list[int] = []
                n_hold = min(500, max(0, pool_size - config.total_budget - 10))
                if n_hold > 0:
                    hrng = np.random.default_rng(12345)
                    holdout = sorted(
                        hrng.choice(pool_size, size=n_hold, replace=False).tolist()
                    )
                    selectable = np.setdiff1d(np.arange(pool_size), np.asarray(holdout))
                    rng = np.random.default_rng(config.loop.seed)
                    initial = sorted(
                        rng.choice(
                            selectable,
                            size=min(config.loop.n_initial, selectable.size),
                            replace=False,
                        ).tolist()
                    )
                else:
                    initial = None

                opt = ClosedLoopOptimizer(
                    assay,
                    config,
                    features=features,
                    zero_shot_scores=features.zero_shot,
                    initial_indices=initial,
                    holdout_indices=holdout,
                )
                session.optimizer = opt
                session.status = "ready"
                self.store.update_experiment(
                    exp_id,
                    status="ready",
                    n_measured=opt.oracle.n_queries,
                    best_measured=opt.best_measured,
                    best_mutant=opt.best_mutant,
                )
                self.store.add_round(exp_id, 0, opt.history[0].to_dict())
                session.emit("ready", self.state(exp_id))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to prepare experiment %s", exp_id)
                session.status = "error"
                session.error = f"{type(exc).__name__}: {exc}"
                self.store.update_experiment(
                    exp_id, status="error", error=session.error
                )
                session.emit("error", {"error": session.error})

        threading.Thread(target=prepare, daemon=True, name=f"prep-{exp_id}").start()
        return session

    def get(self, exp_id: str) -> Session | None:
        return self.sessions.get(exp_id)

    def step(self, exp_id: str) -> dict:
        session = self._require(exp_id)
        with session._lock:
            opt = session.optimizer
            if opt is None:
                raise RuntimeError(f"Experiment is {session.status}, not ready")
            if opt.finished:
                raise RuntimeError("Experiment already finished")
            record = opt.step()
            self.store.add_round(exp_id, record.round, record.to_dict())
            status = "done" if opt.finished else "paused"
            session.status = status
            self.store.update_experiment(
                exp_id,
                status=status,
                round=opt.round,
                n_measured=opt.oracle.n_queries,
                best_measured=opt.best_measured,
                best_mutant=opt.best_mutant,
            )
        payload = record.to_dict()
        session.emit("round", payload)
        session.emit("status", {"status": session.status})
        return payload

    def run(self, exp_id: str) -> None:
        """Run remaining rounds in a background thread."""
        session = self._require(exp_id)
        if session._thread and session._thread.is_alive():
            return
        if session.optimizer is None:
            raise RuntimeError(f"Experiment is {session.status}, not ready")
        session._stop.clear()

        def worker() -> None:
            session.status = "running"
            self.store.update_experiment(exp_id, status="running")
            session.emit("status", {"status": "running"})
            try:
                while not session._stop.is_set():
                    opt = session.optimizer
                    assert opt is not None
                    if opt.finished:
                        break
                    self.step(exp_id)
                    if opt.finished:
                        break
                final = "done" if session.optimizer.finished else "paused"
                session.status = final
                self.store.update_experiment(exp_id, status=final)
                session.emit("status", {"status": final})
                if final == "done":
                    session.emit("finished", self.state(exp_id))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Experiment %s failed", exp_id)
                session.status = "error"
                session.error = f"{type(exc).__name__}: {exc}"
                self.store.update_experiment(
                    exp_id, status="error", error=session.error
                )
                session.emit("error", {"error": session.error})

        session._thread = threading.Thread(
            target=worker, daemon=True, name=f"run-{exp_id}"
        )
        session._thread.start()

    def pause(self, exp_id: str) -> dict:
        session = self._require(exp_id)
        session._stop.set()
        return {"id": exp_id, "status": "pausing"}

    def reset(self, exp_id: str) -> Session:
        """Start a fresh run with the same configuration."""
        session = self._require(exp_id)
        session._stop.set()
        return self.create(session.config)

    def _require(self, exp_id: str) -> Session:
        session = self.sessions.get(exp_id)
        if session is None:
            raise KeyError(exp_id)
        return session

    # ------------------------------------------------------------------

    def state(self, exp_id: str) -> dict:
        session = self._require(exp_id)
        base = {
            "id": exp_id,
            "status": session.status,
            "assay_id": session.config.assay_id,
            "acquisition": session.config.loop.acquisition,
            "round": 0,
            "n_rounds": session.config.loop.n_rounds,
            "n_measured": 0,
            "budget": session.config.total_budget,
            "error": session.error,
            "config": session.config.to_dict(),
        }
        opt = session.optimizer
        if opt is None:
            return base
        n, best = opt.learning_curve()
        base.update(
            {
                "round": opt.round,
                "n_measured": opt.oracle.n_queries,
                "best_measured": opt.best_measured,
                "best_mutant": opt.best_mutant,
                "pool_size": len(opt.pool),
                "surrogate_kind": opt.surrogate.kind,
                "features": opt.features.to_dict(),
                "learning_curve": {
                    "n_measured": [float(v) for v in n.tolist()],
                    "best_measured": best.tolist(),
                },
            }
        )
        return base

    def results(self, exp_id: str) -> dict:
        session = self._require(exp_id)
        state = self.state(exp_id)
        opt = session.optimizer
        out = {
            **{
                k: state.get(k)
                for k in (
                    "id",
                    "status",
                    "assay_id",
                    "acquisition",
                    "n_measured",
                    "budget",
                    "best_measured",
                    "best_mutant",
                    "learning_curve",
                )
            },
            "rounds": self.store.get_rounds(exp_id),
        }
        if opt is not None:
            wt = opt.pool.wild_type
            best_seq = opt.pool.sequences[opt.best_index]
            out["wild_type"] = wt
            out["best_sequence"] = best_seq
            out["pool_stats"] = opt.oracle.ground_truth_summary()
            diffs = []
            for pos in opt.pool.mutated_positions(opt.best_index):
                diffs.append(
                    {
                        "position": pos,
                        "wild_type_aa": wt[pos - 1],
                        "variant_aa": best_seq[pos - 1],
                    }
                )
            out["differences"] = diffs
        return out

    def list_experiments(self, limit: int = 50) -> list[dict]:
        return self.store.list_experiments(limit)

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def start_benchmark(
        self,
        config: ExperimentConfig,
        methods: list[str],
        n_seeds: int,
        holdout_size: int = 500,
    ) -> str:
        bench_id = uuid.uuid4().hex[:12]
        self.store.create_benchmark(bench_id, config.to_dict())
        self.benchmarks[bench_id] = {"progress": 0.0, "status": "running"}

        def worker() -> None:
            try:
                def progress(done, total, method, seed, metrics) -> None:
                    frac = done / max(total, 1)
                    self.benchmarks[bench_id]["progress"] = frac
                    self.store.update_benchmark(bench_id, progress=frac)

                result = run_benchmark(
                    config,
                    methods=methods,
                    seeds=list(range(n_seeds)),
                    holdout_size=holdout_size,
                    progress=progress,
                )
                # API-triggered runs are saved under results/api/<id> so they
                # can never overwrite the curated top-level benchmark files
                # that the README quotes and /api/results/precomputed serves.
                paths = save_benchmark(
                    result, out_dir=RESULTS_DIR / "api", tag=bench_id
                )
                payload = result.to_dict()
                payload["saved_files"] = {k: str(v) for k, v in paths.items()}
                self.store.update_benchmark(
                    bench_id,
                    status="done",
                    progress=1.0,
                    result_json=__import__("json").dumps(payload),
                )
                self.benchmarks[bench_id] = {"progress": 1.0, "status": "done"}
            except Exception as exc:  # noqa: BLE001
                logger.exception("Benchmark %s failed", bench_id)
                self.store.update_benchmark(
                    bench_id, status="error", error=f"{type(exc).__name__}: {exc}"
                )
                self.benchmarks[bench_id] = {"progress": 0.0, "status": "error"}

        threading.Thread(target=worker, daemon=True, name=f"bench-{bench_id}").start()
        return bench_id

    def get_benchmark(self, bench_id: str) -> dict | None:
        return self.store.get_benchmark(bench_id)

    def list_benchmarks(self, limit: int = 20) -> list[dict]:
        return self.store.list_benchmarks(limit)
