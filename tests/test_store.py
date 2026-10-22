"""SQLite persistence for experiments, rounds and benchmarks."""

from __future__ import annotations

import pytest

from drucion.config import ExperimentConfig
from drucion.experiments.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.sqlite")


@pytest.fixture
def config():
    cfg = ExperimentConfig()
    cfg.loop.n_initial = 20
    cfg.loop.batch_size = 5
    cfg.loop.n_rounds = 4
    return cfg


class TestExperiments:
    def test_create_and_read_back(self, store, config):
        store.create_experiment("exp1", config.to_dict(), config.total_budget)
        row = store.get_experiment("exp1")
        assert row is not None
        assert row["id"] == "exp1"
        assert row["assay_id"] == config.assay_id
        assert row["acquisition"] == config.loop.acquisition
        assert row["status"] == "created"
        assert row["budget"] == 40
        assert row["n_rounds"] == 4
        assert row["config"]["loop"]["n_initial"] == 20

    def test_unknown_experiment_returns_none(self, store):
        assert store.get_experiment("nope") is None

    def test_update_fields(self, store, config):
        store.create_experiment("exp1", config.to_dict(), config.total_budget)
        store.update_experiment(
            "exp1", status="done", n_measured=40, best_measured=1.25, best_mutant="K2R"
        )
        row = store.get_experiment("exp1")
        assert row["status"] == "done"
        assert row["n_measured"] == 40
        assert row["best_measured"] == pytest.approx(1.25)
        assert row["best_mutant"] == "K2R"

    def test_update_with_no_fields_is_a_noop(self, store, config):
        store.create_experiment("exp1", config.to_dict(), config.total_budget)
        store.update_experiment("exp1")
        assert store.get_experiment("exp1")["status"] == "created"

    def test_updated_at_advances(self, store, config):
        store.create_experiment("exp1", config.to_dict(), config.total_budget)
        before = store.get_experiment("exp1")["updated_at"]
        store.update_experiment("exp1", status="running")
        assert store.get_experiment("exp1")["updated_at"] >= before

    def test_rounds_round_trip_in_order(self, store, config):
        store.create_experiment("exp1", config.to_dict(), config.total_budget)
        for r in (2, 1, 3):  # inserted out of order on purpose
            store.add_round("exp1", r, {"round": r, "best_measured": float(r)})
        rounds = store.get_rounds("exp1")
        assert [x["round"] for x in rounds] == [1, 2, 3]

    def test_adding_the_same_round_twice_replaces_it(self, store, config):
        store.create_experiment("exp1", config.to_dict(), config.total_budget)
        store.add_round("exp1", 1, {"round": 1, "best_measured": 1.0})
        store.add_round("exp1", 1, {"round": 1, "best_measured": 2.0})
        rounds = store.get_rounds("exp1")
        assert len(rounds) == 1
        assert rounds[0]["best_measured"] == 2.0

    def test_rounds_are_scoped_to_their_experiment(self, store, config):
        store.create_experiment("a", config.to_dict(), config.total_budget)
        store.create_experiment("b", config.to_dict(), config.total_budget)
        store.add_round("a", 1, {"round": 1})
        assert len(store.get_rounds("a")) == 1
        assert store.get_rounds("b") == []

    def test_listing_is_newest_first_and_respects_limit(self, store, config):
        for i in range(5):
            store.create_experiment(f"exp{i}", config.to_dict(), config.total_budget)
        rows = store.list_experiments(limit=3)
        assert len(rows) == 3
        times = [r["created_at"] for r in rows]
        assert times == sorted(times, reverse=True)

    def test_nested_round_payload_survives_json_round_trip(self, store, config):
        store.create_experiment("exp1", config.to_dict(), config.total_budget)
        record = {
            "round": 1,
            "selected": [
                {"mutant": "K2R", "measured_fitness": 1.5, "predicted_fitness": 1.2}
            ],
            "surrogate_info": {"n_train": 20, "noise": 0.01},
        }
        store.add_round("exp1", 1, record)
        assert store.get_rounds("exp1")[0] == record


class TestBenchmarks:
    def test_create_and_read_back(self, store, config):
        store.create_benchmark("b1", config.to_dict())
        row = store.get_benchmark("b1")
        assert row["id"] == "b1"
        assert row["status"] == "running"
        assert row["result"] is None
        assert row["config"]["assay_id"] == config.assay_id

    def test_unknown_benchmark_returns_none(self, store):
        assert store.get_benchmark("nope") is None

    def test_progress_and_result_update(self, store, config):
        import json

        store.create_benchmark("b1", config.to_dict())
        store.update_benchmark("b1", progress=0.5)
        assert store.get_benchmark("b1")["progress"] == pytest.approx(0.5)
        payload = {"methods": ["ucb"], "summary": {"ucb": {"best_measured_mean": 1.0}}}
        store.update_benchmark(
            "b1", status="done", progress=1.0, result_json=json.dumps(payload)
        )
        row = store.get_benchmark("b1")
        assert row["status"] == "done"
        assert row["result"] == payload

    def test_error_is_recorded(self, store, config):
        store.create_benchmark("b1", config.to_dict())
        store.update_benchmark("b1", status="error", error="boom")
        row = store.get_benchmark("b1")
        assert row["status"] == "error"
        assert row["error"] == "boom"

    def test_listing(self, store, config):
        for i in range(3):
            store.create_benchmark(f"b{i}", config.to_dict())
        rows = store.list_benchmarks(limit=2)
        assert len(rows) == 2


class TestPersistence:
    def test_data_survives_a_new_connection(self, tmp_path, config):
        path = tmp_path / "persist.sqlite"
        s1 = Store(path)
        s1.create_experiment("exp1", config.to_dict(), config.total_budget)
        s1.add_round("exp1", 1, {"round": 1, "best_measured": 3.0})
        s1.update_experiment("exp1", status="done", best_measured=3.0)

        s2 = Store(path)
        row = s2.get_experiment("exp1")
        assert row["status"] == "done"
        assert row["best_measured"] == pytest.approx(3.0)
        assert s2.get_rounds("exp1")[0]["best_measured"] == 3.0

    def test_schema_creation_is_idempotent(self, tmp_path, config):
        path = tmp_path / "idem.sqlite"
        Store(path).create_experiment("exp1", config.to_dict(), 10)
        # Re-running the schema script must not wipe or fail.
        assert Store(path).get_experiment("exp1") is not None
