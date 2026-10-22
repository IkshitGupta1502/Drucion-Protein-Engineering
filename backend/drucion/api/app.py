"""FastAPI application for Drucion.

The API is a thin shell: it validates input, owns lifecycle, and streams
events. Every scientific decision lives in :mod:`drucion.optimization`,
:mod:`drucion.models` and :mod:`drucion.data`.
"""

from __future__ import annotations

import json
import logging
import queue
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from ..config import REPO_ROOT, RESULTS_DIR, EmbeddingConfig, ExperimentConfig
from ..data.proteingym import (
    AssayValidationError,
    list_assays,
    load_assay,
    load_assay_metadata,
)
from ..data.structures import (
    StructureMisaligned,
    StructureUnavailable,
    has_structure,
    load_structure,
)
from ..experiments.worklist import WorklistError
from .schemas import (
    AssayDetail,
    AssaySummary,
    BenchmarkOut,
    BenchmarkRequest,
    ExperimentOut,
    ExperimentRequest,
    ExperimentResults,
)
from .sessions import SessionManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Drucion",
    version="0.1.0",
    description=(
        "Closed-loop, uncertainty-aware active learning for protein "
        "engineering, evaluated retrospectively against real ProteinGym DMS "
        "measurements."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = SessionManager()


def _to_config(req: ExperimentRequest) -> ExperimentConfig:
    cfg = ExperimentConfig(assay_id=req.assay_id, pool_size=req.pool_size)
    cfg.embedding = EmbeddingConfig(
        featurizer=req.featurizer, n_components=req.n_components
    )
    cfg.surrogate.kind = req.surrogate
    cfg.loop.n_initial = req.loop.n_initial
    cfg.loop.batch_size = req.loop.batch_size
    cfg.loop.n_rounds = req.loop.n_rounds
    cfg.loop.seed = req.loop.seed
    cfg.loop.acquisition = req.loop.acquisition
    cfg.loop.ucb_beta = req.loop.ucb_beta
    cfg.loop.diversity_quantile = req.loop.diversity_quantile
    return cfg


# ---------------------------------------------------------------------------
# Assays
# ---------------------------------------------------------------------------


@app.get("/api/assays", response_model=list[AssaySummary])
def get_assays() -> list[dict]:
    """Every ProteinGym substitution assay available on disk."""
    try:
        return list_assays()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/assays/{assay_id}", response_model=AssayDetail)
def get_assay(assay_id: str, full: bool = False) -> dict:
    """Metadata and provenance for one assay.

    ``full=true`` additionally loads and validates the CSV, returning the
    validation report, pool size and pool-level score statistics. Those
    statistics are shown for context only — a running optimizer never sees
    them.
    """
    try:
        meta = load_assay_metadata(assay_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown assay: {assay_id}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    out = meta.to_dict()
    if full:
        try:
            assay = load_assay(assay_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except AssayValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        out["validation"] = assay.validation
        out["pool_size"] = len(assay.pool)
        out["pool_stats"] = assay.oracle.ground_truth_summary()
    out["has_structure"] = has_structure(meta.uniprot_id)
    return out


@app.get("/api/assays/{assay_id}/structure")
def get_assay_structure(assay_id: str) -> Response:
    """Wild-type PDB structure for the 3D explorer.

    This is the published AlphaFold2 model for the target protein, served
    verbatim. There is no structure for any mutant — a single substitution
    does not visibly change a backbone, so the explorer marks mutated
    positions on the wild-type fold rather than inventing mutant geometry.

    The endpoint refuses (409) if the structure's residue numbering does not
    match the assay's wild-type sequence, since highlighting position N on a
    structure whose residue N is a different amino acid would mislead.
    """
    try:
        structure = load_structure(assay_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown assay: {assay_id}") from exc
    except StructureUnavailable as exc:
        raise HTTPException(404, str(exc)) from exc
    except StructureMisaligned as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(
        content=structure.pdb_text,
        media_type="chemical/x-pdb",
        headers={
            "X-Structure-Source": structure.source,
            "X-Structure-Residues": str(structure.n_residues),
            "Cache-Control": "public, max-age=86400",
        },
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@app.post("/api/experiments", response_model=ExperimentOut, status_code=201)
def create_experiment(req: ExperimentRequest) -> dict:
    """Create an experiment. Feature computation runs in the background."""
    cfg = _to_config(req)
    try:
        load_assay_metadata(cfg.assay_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown assay: {cfg.assay_id}") from exc
    if cfg.total_budget >= 200_000:
        raise HTTPException(422, "Requested budget is implausibly large")
    session = manager.create(cfg)
    return manager.state(session.id)


@app.get("/api/experiments", response_model=list[dict])
def list_experiments(limit: int = 50) -> list[dict]:
    return manager.list_experiments(limit)


@app.get("/api/experiments/{exp_id}", response_model=ExperimentOut)
def get_experiment(exp_id: str) -> dict:
    try:
        return manager.state(exp_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc


@app.post("/api/experiments/{exp_id}/step", response_model=dict)
def step_experiment(exp_id: str) -> dict:
    """Run exactly one experimental round, so users can inspect each change."""
    try:
        return manager.step(exp_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/experiments/{exp_id}/propose", response_model=dict)
def propose_batch(exp_id: str) -> dict:
    """Choose the next batch to test, without revealing any measurement.

    This is the first half of a real campaign: Drucion commits to a batch,
    you take it to the bench, and the values come back through
    ``/measurements``. Calling it twice returns the same pending batch.
    """
    try:
        return manager.propose(exp_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/experiments/{exp_id}/extend", response_model=dict)
def extend_experiment(exp_id: str, additional_rounds: int = 1) -> dict:
    """Allow more rounds than the run was originally planned for.

    The fixed round budget exists so benchmark comparisons are fair. A real
    campaign has no such stopping rule, so extending is explicit and recorded
    rather than silent.
    """
    if not 1 <= additional_rounds <= 100:
        raise HTTPException(422, "additional_rounds must be between 1 and 100")
    try:
        return manager.extend(exp_id, additional_rounds)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/experiments/{exp_id}/worklist.csv")
def download_worklist(exp_id: str, prefill: bool = False) -> Response:
    """Download the pending batch as a CSV worklist to take to the bench.

    ``prefill=true`` returns the sheet already filled in with the **real
    published measurements** for those variants, so the upload path can be
    exercised end to end without inventing data. The file states that
    provenance in its header.
    """
    try:
        text, filename = manager.worklist_csv(exp_id, prefill=prefill)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return Response(
        content=text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/experiments/{exp_id}/measurements", response_model=dict)
async def submit_measurements(
    exp_id: str, file: UploadFile | None = None, request: Request = None
) -> dict:
    """Upload measured results for the pending batch and re-optimize.

    Accepts the filled-in worklist either as a multipart file upload or as a
    raw CSV body. Every row that could not be used is reported back with the
    reason; nothing is silently dropped. Values arriving here are recorded
    with ``source="uploaded"`` and are never presented as ProteinGym data.
    """
    if file is not None:
        raw = await file.read()
    else:
        raw = await request.body()
    if not raw:
        raise HTTPException(422, "No file content received")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(422, f"File is not valid UTF-8 text: {exc}") from exc

    try:
        return manager.submit_measurements(exp_id, text)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc
    except WorklistError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/experiments/{exp_id}/run", response_model=ExperimentOut)
def run_experiment(exp_id: str) -> dict:
    try:
        manager.run(exp_id)
        return manager.state(exp_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/experiments/{exp_id}/pause", response_model=dict)
def pause_experiment(exp_id: str) -> dict:
    try:
        return manager.pause(exp_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc


@app.post("/api/experiments/{exp_id}/reset", response_model=ExperimentOut)
def reset_experiment(exp_id: str) -> dict:
    try:
        session = manager.reset(exp_id)
        return manager.state(session.id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc


@app.get("/api/experiments/{exp_id}/results", response_model=ExperimentResults)
def experiment_results(exp_id: str) -> dict:
    try:
        return manager.results(exp_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc


@app.get("/api/experiments/{exp_id}/stream")
async def stream_experiment(exp_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events: status, embedding progress and each round as it lands."""
    try:
        session = manager._require(exp_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown experiment: {exp_id}") from exc

    q = session.subscribe()

    async def event_source():
        try:
            # Replay current state so a late subscriber is never out of sync.
            yield _sse("status", {"status": session.status})
            snapshot = manager.state(exp_id)
            yield _sse("snapshot", snapshot)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = q.get(timeout=0.5)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse(payload["event"], payload["data"])
                if payload["event"] in ("finished", "error"):
                    break
        finally:
            session.unsubscribe(q)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@app.post("/api/benchmarks", response_model=BenchmarkOut, status_code=201)
def create_benchmark(req: BenchmarkRequest) -> dict:
    cfg = ExperimentConfig(assay_id=req.assay_id, pool_size=req.pool_size)
    cfg.embedding = EmbeddingConfig(featurizer=req.featurizer)
    cfg.surrogate.kind = req.surrogate
    cfg.loop.n_initial = req.loop.n_initial
    cfg.loop.batch_size = req.loop.batch_size
    cfg.loop.n_rounds = req.loop.n_rounds
    cfg.loop.ucb_beta = req.loop.ucb_beta
    cfg.loop.diversity_quantile = req.loop.diversity_quantile
    try:
        load_assay_metadata(cfg.assay_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown assay: {cfg.assay_id}") from exc

    bench_id = manager.start_benchmark(
        cfg, list(req.methods), req.n_seeds, req.holdout_size
    )
    return {"id": bench_id, "status": "running", "assay_id": cfg.assay_id}


@app.get("/api/benchmarks", response_model=list[dict])
def list_benchmarks(limit: int = 20) -> list[dict]:
    return manager.list_benchmarks(limit)


@app.get("/api/benchmarks/{bench_id}", response_model=BenchmarkOut)
def get_benchmark(bench_id: str) -> dict:
    row = manager.get_benchmark(bench_id)
    if row is None:
        raise HTTPException(404, f"Unknown benchmark: {bench_id}")
    return {
        "id": row["id"],
        "status": row["status"],
        "assay_id": row["assay_id"],
        "progress": row.get("progress") or 0.0,
        "result": row.get("result"),
        "error": row.get("error"),
    }


@app.get("/api/results/precomputed")
def precomputed_results() -> dict:
    """Benchmark JSON files already on disk, for instant dashboard load.

    Lets the demo show real saved results without re-running anything.
    """
    out = {}
    results_dir = Path(RESULTS_DIR)
    if results_dir.exists():
        for path in sorted(results_dir.glob("*_benchmark.json")):
            try:
                out[path.stem.replace("_benchmark", "")] = json.loads(
                    path.read_text()
                )
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable result file %s", path)
    return {"benchmarks": out}


@app.get("/api/health")
def health() -> dict:
    import torch

    from ..config import resolve_device

    return {
        "status": "ok",
        "device": resolve_device(),
        "torch": torch.__version__,
        "n_assays_available": len(list_assays()) if _assays_ok() else 0,
    }


def _assays_ok() -> bool:
    try:
        list_assays()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Static frontend (built Vite bundle), mounted last so /api wins.
# ---------------------------------------------------------------------------

_DIST = Path(REPO_ROOT) / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        candidate = _DIST / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")

else:

    @app.get("/")
    def index_missing() -> JSONResponse:
        return JSONResponse(
            {
                "message": (
                    "Drucion API is running, but the frontend bundle is not "
                    "built. Run `npm install && npm run build` in frontend/, "
                    "or use the Vite dev server on port 5173."
                ),
                "docs": "/docs",
                "health": "/api/health",
            }
        )
