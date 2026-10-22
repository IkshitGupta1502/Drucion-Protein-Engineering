"""API contract tests.

These run against the real assay directory when it is present (the endpoints
read ProteinGym metadata), but never start a long optimization: experiment
creation is exercised with a tiny pool and the ``onehot`` featurizer so no
ESM-2 weights are needed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from drucion.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthAndAssays:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["device"] in ("cpu", "cuda", "mps")

    def test_list_assays(self, client, require_real_data):
        r = client.get("/api/assays")
        assert r.status_code == 200
        assays = r.json()
        assert len(assays) > 100
        ids = {a["assay_id"] for a in assays}
        assert "A4GRB6_PSEAI_Chen_2020" in ids
        first = assays[0]
        assert {"assay_id", "molecule_name", "seq_len", "available"} <= set(first)

    def test_assay_detail(self, client, require_real_data):
        r = client.get("/api/assays/A4GRB6_PSEAI_Chen_2020")
        assert r.status_code == 200
        body = r.json()
        assert body["molecule_name"] == "Beta-lactamase VIM-2"
        assert len(body["wild_type"]) == body["seq_len"]
        assert body["source"].startswith("ProteinGym")

    def test_assay_detail_full_includes_validation(self, client, require_real_data):
        r = client.get("/api/assays/A4GRB6_PSEAI_Chen_2020?full=true")
        assert r.status_code == 200
        body = r.json()
        assert body["pool_size"] > 4000
        assert body["validation"]["pool_size"] == body["pool_size"]
        assert "pool_max" in body["pool_stats"]

    def test_unknown_assay_is_404(self, client):
        assert client.get("/api/assays/NOT_REAL").status_code == 404


class TestValidation:
    def test_rejects_empty_assay_id(self, client):
        r = client.post("/api/experiments", json={"assay_id": "   "})
        assert r.status_code == 422

    @pytest.mark.parametrize(
        "loop",
        [
            {"n_initial": 0},
            {"n_initial": 1},
            {"batch_size": 0},
            {"n_rounds": 0},
            {"n_rounds": 500},
            {"acquisition": "not_a_method"},
            {"ucb_beta": -1.0},
            {"diversity_quantile": 0.9},
            {"seed": -1},
        ],
    )
    def test_rejects_out_of_range_loop_settings(self, client, loop):
        r = client.post(
            "/api/experiments",
            json={"assay_id": "A4GRB6_PSEAI_Chen_2020", "loop": loop},
        )
        assert r.status_code == 422, loop

    def test_rejects_bad_featurizer(self, client):
        r = client.post(
            "/api/experiments",
            json={"assay_id": "A4GRB6_PSEAI_Chen_2020", "featurizer": "nope"},
        )
        assert r.status_code == 422

    def test_unknown_assay_on_create_is_404(self, client):
        r = client.post("/api/experiments", json={"assay_id": "NOT_REAL"})
        assert r.status_code == 404

    def test_benchmark_requires_at_least_one_method(self, client):
        r = client.post(
            "/api/benchmarks",
            json={"assay_id": "A4GRB6_PSEAI_Chen_2020", "methods": []},
        )
        assert r.status_code == 422

    def test_benchmark_rejects_unknown_method(self, client):
        r = client.post(
            "/api/benchmarks",
            json={"assay_id": "A4GRB6_PSEAI_Chen_2020", "methods": ["telepathy"]},
        )
        assert r.status_code == 422


class TestMissingResources:
    def test_unknown_experiment_endpoints_are_404(self, client):
        for path, method in [
            ("/api/experiments/deadbeef", "get"),
            ("/api/experiments/deadbeef/results", "get"),
            ("/api/experiments/deadbeef/step", "post"),
            ("/api/experiments/deadbeef/run", "post"),
            ("/api/experiments/deadbeef/pause", "post"),
        ]:
            r = getattr(client, method)(path)
            assert r.status_code == 404, path

    def test_unknown_benchmark_is_404(self, client):
        assert client.get("/api/benchmarks/deadbeef").status_code == 404


@pytest.mark.real_data
class TestExperimentLifecycle:
    """Create -> ready -> step -> results, on a small real pool."""

    PAYLOAD = {
        "assay_id": "A4GRB6_PSEAI_Chen_2020",
        "pool_size": 300,
        "featurizer": "onehot",
        "n_components": 8,
        "loop": {
            "n_initial": 20,
            "batch_size": 5,
            "n_rounds": 3,
            "acquisition": "ucb",
            "seed": 0,
        },
    }

    def _wait_ready(self, client, exp_id, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            body = client.get(f"/api/experiments/{exp_id}").json()
            if body["status"] in ("ready", "error", "done", "paused"):
                return body
            time.sleep(0.2)
        pytest.fail(f"Experiment {exp_id} never became ready")

    @pytest.fixture
    def experiment(self, client, require_real_data):
        r = client.post("/api/experiments", json=self.PAYLOAD)
        assert r.status_code == 201, r.text
        exp_id = r.json()["id"]
        body = self._wait_ready(client, exp_id)
        assert body["status"] == "ready", body.get("error")
        return exp_id

    def test_created_experiment_reports_budget_and_initial_measurements(
        self, client, experiment
    ):
        body = client.get(f"/api/experiments/{experiment}").json()
        assert body["budget"] == 20 + 5 * 3
        assert body["n_measured"] == 20
        assert body["round"] == 0
        assert body["pool_size"] == 300
        assert body["best_measured"] is not None

    def test_step_advances_one_round(self, client, experiment):
        before = client.get(f"/api/experiments/{experiment}").json()
        r = client.post(f"/api/experiments/{experiment}/step")
        assert r.status_code == 200, r.text
        record = r.json()
        assert record["round"] == before["round"] + 1
        assert len(record["selected"]) == 5
        assert record["n_measured"] == before["n_measured"] + 5
        for v in record["selected"]:
            assert "measured_fitness" in v
            assert "predicted_fitness" in v
            assert "predicted_uncertainty" in v
            assert "acquisition_score" in v

    def test_stepping_past_the_end_is_409(self, client, experiment):
        for _ in range(3):
            assert client.post(f"/api/experiments/{experiment}/step").status_code == 200
        r = client.post(f"/api/experiments/{experiment}/step")
        assert r.status_code == 409

    def test_results_expose_wild_type_diff_and_curve(self, client, experiment):
        client.post(f"/api/experiments/{experiment}/step")
        r = client.get(f"/api/experiments/{experiment}/results")
        assert r.status_code == 200
        body = r.json()
        assert body["wild_type"]
        assert body["best_sequence"]
        estimate = body["budget_estimate"]
        assert estimate is not None
        assert estimate["target_score"] == body["best_measured"]
        assert estimate["actual_measurements"] <= body["n_measured"]
        assert estimate["random_expected_measurements"] > 0
        assert len(body["wild_type"]) == len(body["best_sequence"])
        assert body["differences"]
        for d in body["differences"]:
            pos = d["position"]
            assert body["wild_type"][pos - 1] == d["wild_type_aa"]
            assert body["best_sequence"][pos - 1] == d["variant_aa"]
            assert d["wild_type_aa"] != d["variant_aa"]
        curve = body["learning_curve"]
        assert len(curve["n_measured"]) == len(curve["best_measured"])
        assert curve["best_measured"] == sorted(curve["best_measured"])

    def test_run_to_completion(self, client, experiment):
        assert client.post(f"/api/experiments/{experiment}/run").status_code == 200
        deadline = time.time() + 180
        while time.time() < deadline:
            body = client.get(f"/api/experiments/{experiment}").json()
            if body["status"] in ("done", "error"):
                break
            time.sleep(0.25)
        assert body["status"] == "done", body.get("error")
        assert body["n_measured"] == body["budget"] == 35
        assert body["round"] == 3

    def test_experiment_appears_in_listing(self, client, experiment):
        rows = client.get("/api/experiments").json()
        assert experiment in {r["id"] for r in rows}

    def test_reset_starts_a_fresh_run(self, client, experiment):
        client.post(f"/api/experiments/{experiment}/step")
        r = client.post(f"/api/experiments/{experiment}/reset")
        assert r.status_code == 200
        fresh = r.json()
        assert fresh["id"] != experiment
        body = self._wait_ready(client, fresh["id"])
        assert body["round"] == 0
        assert body["n_measured"] == 20

    def test_stream_emits_snapshot(self, client, experiment):
        with client.stream(
            "GET", f"/api/experiments/{experiment}/stream"
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            events = []
            for line in response.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
                if len(events) >= 2:
                    break
            assert "status" in events
            assert "snapshot" in events


@pytest.mark.real_data
class TestBenchmarkEndpoint:
    """A tiny but complete benchmark through the API."""

    def test_benchmark_runs_and_returns_comparable_results(
        self, client, require_real_data
    ):
        r = client.post(
            "/api/benchmarks",
            json={
                "assay_id": "A4GRB6_PSEAI_Chen_2020",
                "methods": ["random", "ucb"],
                "n_seeds": 2,
                "pool_size": 250,
                "featurizer": "onehot",
                "holdout_size": 40,
                "loop": {"n_initial": 20, "batch_size": 5, "n_rounds": 2},
            },
        )
        assert r.status_code == 201, r.text
        bench_id = r.json()["id"]

        deadline = time.time() + 300
        body = None
        while time.time() < deadline:
            body = client.get(f"/api/benchmarks/{bench_id}").json()
            if body["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert body is not None and body["status"] == "done", body

        result = body["result"]
        assert set(result["methods"]) == {"random", "ucb"}
        assert result["seeds"] == [0, 1]

        # Fairness contract: every run consumed the same budget.
        budget = 20 + 5 * 2
        assert {run["n_measured"] for run in result["runs"]} == {budget}
        assert len(result["runs"]) == 4  # 2 methods x 2 seeds

        # Curves share one budget grid, so they are directly comparable.
        grids = {tuple(c["n_measured"]) for c in result["curves"].values()}
        assert len(grids) == 1
        assert list(grids)[0] == (20, 25, 30)

        # Best-so-far must never decrease.
        for curve in result["curves"].values():
            assert curve["mean"] == sorted(curve["mean"])

        for method in ("random", "ucb"):
            s = result["summary"][method]
            assert s["n_seeds"] == 2
            assert np.isfinite(s["best_measured_mean"])
        assert "paired" in result and "efficiency" in result

    def test_benchmark_appears_in_listing(self, client, require_real_data):
        rows = client.get("/api/benchmarks").json()
        assert isinstance(rows, list)

    def test_api_runs_never_overwrite_curated_results(
        self, client, require_real_data
    ):
        """A benchmark launched from the UI must not clobber results/.

        The top-level ``results/*_benchmark.json`` files are the curated runs
        the README quotes and the dashboard loads. An API-triggered run — or a
        test of one — writing there would silently replace a 10-seed benchmark
        with whatever small job happened to run last.
        """
        from drucion.config import RESULTS_DIR

        curated = sorted(RESULTS_DIR.glob("*_benchmark.json"))
        before = {p: p.stat().st_mtime for p in curated}

        r = client.post(
            "/api/benchmarks",
            json={
                "assay_id": "A4GRB6_PSEAI_Chen_2020",
                "methods": ["random"],
                "n_seeds": 1,
                "pool_size": 150,
                "featurizer": "onehot",
                "holdout_size": 20,
                "loop": {"n_initial": 10, "batch_size": 5, "n_rounds": 1},
            },
        )
        assert r.status_code == 201
        bench_id = r.json()["id"]
        deadline = time.time() + 180
        while time.time() < deadline:
            body = client.get(f"/api/benchmarks/{bench_id}").json()
            if body["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert body["status"] == "done", body

        for path, mtime in before.items():
            assert path.exists(), f"{path.name} was deleted by an API run"
            assert path.stat().st_mtime == mtime, (
                f"{path.name} was overwritten by an API-triggered benchmark"
            )
        saved = body["result"]["saved_files"]["json"]
        assert "/api/" in saved, saved
