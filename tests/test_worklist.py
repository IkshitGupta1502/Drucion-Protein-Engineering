"""Worklist export and bench-result intake.

Intake is the one place where numbers enter Drucion from outside, so it is
strict about what it accepts and loud about what it rejects. These tests pin
that behaviour, and pin the provenance tagging that keeps uploaded values from
being mistaken for published measurements.
"""

from __future__ import annotations

import pytest

from drucion.config import ExperimentConfig
from drucion.data.proteingym import AssayMetadata, load_assay
from drucion.experiments.worklist import (
    EXPORT_COLUMNS,
    IntakeResult,
    WorklistError,
    batch_token,
    export_worklist,
    parse_measurements,
)
from drucion.optimization.loop import ClosedLoopOptimizer, PendingBatch


@pytest.fixture
def batch():
    return PendingBatch(
        round=3,
        indices=[10, 11, 12],
        mutants=["K2R", "T3G", "A4V"],
        sequences=["MRTAY", "MKGAY", "MKTVY"],
        positions=[[2], [3], [4]],
        predicted_fitness=[1.5, -0.2, None],
        predicted_uncertainty=[0.3, 0.4, None],
        acquisition_score=[2.1, 0.6, 0.1],
        surrogate_kind="gp",
    )


@pytest.fixture
def metadata():
    return AssayMetadata(
        assay_id="TEST_ASSAY",
        uniprot_id="TEST",
        molecule_name="Test protein",
        source_organism="in silico",
        selection_assay="synthetic",
        selection_type="Synthetic",
        first_author="Fixture",
        title="t",
        year="2026",
        seq_len=5,
        n_mutants=3,
        n_single_mutants=3,
        n_multiple_mutants=0,
        wild_type="MKTAY",
    )


class TestBatchToken:
    def test_is_stable(self, batch):
        assert batch_token("e1", batch) == batch_token("e1", batch)

    def test_differs_by_experiment(self, batch):
        assert batch_token("e1", batch) != batch_token("e2", batch)

    def test_differs_by_round(self, batch):
        other = PendingBatch(**{**batch.__dict__, "round": 4})
        assert batch_token("e1", batch) != batch_token("e1", other)

    def test_differs_by_contents(self, batch):
        other = PendingBatch(**{**batch.__dict__, "mutants": ["K2R", "T3G", "A4L"]})
        assert batch_token("e1", batch) != batch_token("e1", other)


class TestExport:
    def test_has_header_block_and_rows(self, batch, metadata):
        text = export_worklist("e1", batch, metadata)
        assert "# Drucion worklist" in text
        assert "# round: 3" in text
        assert f"# batch_token: {batch_token('e1', batch)}" in text
        assert ",".join(EXPORT_COLUMNS) in text
        for m in batch.mutants:
            assert m in text

    def test_measured_column_blank_by_default(self, batch, metadata):
        result = parse_measurements(export_worklist("e1", batch, metadata))
        assert result.measurements == {}
        assert sorted(result.skipped) == sorted(batch.mutants)

    def test_prefilled_values_round_trip(self, batch, metadata):
        text = export_worklist(
            "e1", batch, metadata, prefilled=[1.0, 2.0, 3.0], prefill_note="real values"
        )
        assert "# PRE-FILLED: real values" in text
        result = parse_measurements(text, expected=batch.mutants)
        assert result.measurements == {"K2R": 1.0, "T3G": 2.0, "A4V": 3.0}

    def test_missing_prediction_renders_blank_not_zero(self, batch, metadata):
        """A None prediction must not become a misleading 0.0 on the sheet."""
        text = export_worklist("e1", batch, metadata)
        row = [l for l in text.splitlines() if l.startswith("A4V")][0]
        assert ",," in row

    def test_decomposes_the_mutation(self, batch, metadata):
        text = export_worklist("e1", batch, metadata)
        row = [l for l in text.splitlines() if l.startswith("K2R")][0].split(",")
        assert row[1] == "2"
        assert row[2] == "K"
        assert row[3] == "R"


class TestIntakeHappyPath:
    def _sheet(self, rows: str, token: str = "abc123") -> str:
        return f"# batch_token: {token}\nmutant,measured_fitness\n{rows}"

    def test_reads_values(self):
        r = parse_measurements(self._sheet("K2R,1.5\nT3G,-0.25\n"))
        assert r.measurements == {"K2R": 1.5, "T3G": -0.25}
        assert r.n_accepted == 2

    def test_blank_rows_are_skipped_not_rejected(self):
        r = parse_measurements(self._sheet("K2R,1.5\nT3G,\n"))
        assert r.measurements == {"K2R": 1.5}
        assert r.skipped == ["T3G"]
        assert r.rejected == []

    def test_ignores_extra_columns(self):
        text = (
            "# batch_token: abc123\n"
            "mutant,predicted_fitness,measured_fitness,notes\n"
            "K2R,0.9,1.5,looked fine\n"
        )
        assert parse_measurements(text).measurements == {"K2R": 1.5}

    def test_tolerates_whitespace_and_case_in_headers(self):
        text = "mutant, Measured_Fitness \n K2R , 1.5 \n"
        assert parse_measurements(text).measurements == {"K2R": 1.5}

    def test_reports_missing_proposed_variants(self):
        r = parse_measurements(
            self._sheet("K2R,1.5\n"), expected=["K2R", "T3G", "A4V"]
        )
        assert r.missing == ["A4V", "T3G"]

    def test_negative_and_exponent_values_accepted(self):
        r = parse_measurements(self._sheet("K2R,-3.2\nT3G,1e-3\n"))
        assert r.measurements["K2R"] == -3.2
        assert r.measurements["T3G"] == pytest.approx(0.001)


class TestIntakeRejections:
    def _sheet(self, rows: str) -> str:
        return f"mutant,measured_fitness\n{rows}"

    def test_non_numeric_value_rejected_with_reason(self):
        r = parse_measurements(self._sheet("K2R,not_a_number\n"))
        assert r.measurements == {}
        assert r.rejected[0][0] == "K2R"
        assert "not a number" in r.rejected[0][1]

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_non_finite_values_rejected(self, bad):
        r = parse_measurements(self._sheet(f"K2R,{bad}\n"))
        assert r.measurements == {}
        assert "finite" in r.rejected[0][1].lower()

    def test_unparseable_mutation_rejected(self):
        r = parse_measurements(self._sheet("not_a_mutation,1.0\n"))
        assert "unparseable" in r.rejected[0][1]

    def test_blank_mutant_rejected(self):
        r = parse_measurements(self._sheet(",1.0\n"))
        assert "no mutant" in r.rejected[0][1]

    def test_duplicate_rows_rejected(self):
        r = parse_measurements(self._sheet("K2R,1.0\nK2R,2.0\n"))
        assert r.measurements == {"K2R": 1.0}
        assert "duplicate" in r.rejected[0][1]

    def test_variant_outside_the_batch_rejected(self):
        """A value for something nobody proposed has nowhere to go."""
        r = parse_measurements(self._sheet("K2R,1.0\nW9Y,2.0\n"), expected=["K2R"])
        assert r.measurements == {"K2R": 1.0}
        assert any("not part of the batch" in why for _, why in r.rejected)

    def test_missing_required_column_raises(self):
        with pytest.raises(WorklistError, match="measured_fitness"):
            parse_measurements("mutant,notes\nK2R,hello\n")

    def test_empty_file_raises(self):
        with pytest.raises(WorklistError, match="no rows"):
            parse_measurements("# only a comment\n")

    def test_stale_sheet_from_another_batch_is_refused(self):
        text = "# batch_token: OLD\nmutant,measured_fitness\nK2R,1.0\n"
        with pytest.raises(WorklistError, match="fresh worklist"):
            parse_measurements(text, expected_token="NEW")

    def test_token_check_can_be_relaxed(self):
        text = "# batch_token: OLD\nmutant,measured_fitness\nK2R,1.0\n"
        r = parse_measurements(text, expected_token="NEW", strict_token=False)
        assert r.measurements == {"K2R": 1.0}

    def test_sheet_without_token_is_accepted(self):
        """A hand-made sheet is fine; the token guard only fires on mismatch."""
        r = parse_measurements(
            "mutant,measured_fitness\nK2R,1.0\n", expected_token="NEW"
        )
        assert r.measurements == {"K2R": 1.0}


class TestIntakeResult:
    def test_serialises_every_category(self):
        r = IntakeResult(
            measurements={"K2R": 1.0},
            skipped=["T3G"],
            rejected=[("A4V", "bad")],
            missing=["W9Y"],
            token_in_file="tok",
        )
        d = r.to_dict()
        assert d["n_accepted"] == 1
        assert d["n_skipped"] == 1
        assert d["n_rejected"] == 1
        assert d["n_missing"] == 1
        assert d["rejected"] == [{"row": "A4V", "reason": "bad"}]


@pytest.mark.real_data
class TestProspectiveLoop:
    """propose -> export -> ingest -> propose again, on a real assay."""

    @pytest.fixture
    def optimizer(self, require_real_data):
        from drucion.embeddings.features import build_features

        cfg = ExperimentConfig()
        cfg.loop.n_initial = 40
        cfg.loop.batch_size = 5
        cfg.loop.n_rounds = 3
        cfg.embedding.featurizer = "onehot"
        cfg.embedding.include_zero_shot = False
        cfg.pool_size = 400
        assay = load_assay(cfg.assay_id, pool_size=cfg.pool_size)
        features = build_features(assay.pool, cfg.embedding)
        return ClosedLoopOptimizer(assay, cfg, features=features), assay

    def test_propose_does_not_measure(self, optimizer):
        opt, _ = optimizer
        before = opt.n_measured
        batch = opt.propose()
        assert len(batch) == 5
        assert opt.n_measured == before
        assert opt.awaiting_measurements

    def test_propose_is_idempotent(self, optimizer):
        opt, _ = optimizer
        assert opt.propose().mutants == opt.propose().mutants

    def test_proposed_variants_are_unmeasured(self, optimizer):
        opt, _ = optimizer
        batch = opt.propose()
        assert not (set(batch.indices) & set(opt.observed))

    def test_ingest_folds_values_in(self, optimizer):
        opt, assay = optimizer
        batch = opt.propose()
        values = {m: 1.0 + i for i, m in enumerate(batch.mutants)}
        record = opt.ingest(values, source="uploaded")
        assert record.round == 1
        assert opt.n_measured == 45
        assert record.measurement_source == "uploaded"
        assert opt.best_measured == pytest.approx(5.0)

    def test_partial_results_return_variants_to_the_pool(self, optimizer):
        """A failed well should not burn a candidate."""
        opt, _ = optimizer
        batch = opt.propose()
        measured = {batch.mutants[0]: 1.0, batch.mutants[1]: 2.0}
        record = opt.ingest(measured, source="uploaded")
        assert record.n_unmeasured == 3
        assert opt.n_measured == 42
        for idx in batch.indices[2:]:
            assert opt.available[idx]

    def test_next_batch_can_reselect_an_unmeasured_variant(self, optimizer):
        opt, _ = optimizer
        first = opt.propose()
        opt.ingest({first.mutants[0]: 1.0}, source="uploaded")
        second = opt.propose()
        assert second.round == 2
        assert first.indices[0] not in second.indices

    def test_ingest_without_propose_raises(self, optimizer):
        opt, _ = optimizer
        with pytest.raises(RuntimeError, match="No pending batch"):
            opt.ingest({"K2R": 1.0})

    def test_ingest_rejects_unknown_mutant(self, optimizer):
        opt, _ = optimizer
        opt.propose()
        with pytest.raises(ValueError, match="not in the pending batch"):
            opt.ingest({"Z999Z": 1.0})

    def test_ingest_rejects_non_finite(self, optimizer):
        opt, _ = optimizer
        batch = opt.propose()
        with pytest.raises(ValueError, match="Non-finite"):
            opt.ingest({batch.mutants[0]: float("nan")})

    def test_ingest_rejects_empty(self, optimizer):
        opt, _ = optimizer
        opt.propose()
        with pytest.raises(ValueError, match="No usable measurements"):
            opt.ingest({})

    def test_full_csv_round_trip_with_real_values(self, optimizer):
        """The end-to-end path the demo sheet exercises."""
        from drucion.data.proteingym import load_assay_metadata

        opt, assay = optimizer
        batch = opt.propose()
        meta = load_assay_metadata(opt.pool.assay_id)
        real = assay.oracle.peek_for_evaluation(batch.indices).tolist()
        text = export_worklist("e1", batch, meta, prefilled=real, prefill_note="real")

        parsed = parse_measurements(
            text, expected=batch.mutants, expected_token=batch_token("e1", batch)
        )
        assert parsed.n_accepted == len(batch)
        assert parsed.rejected == []

        record = opt.ingest(parsed.measurements, source="uploaded")
        for v in record.selected:
            slot = batch.mutants.index(v.mutant)
            assert v.measured_fitness == pytest.approx(real[slot])

    def test_step_still_works_and_is_tagged_proteingym(self, optimizer):
        """The retrospective path must be unchanged by the refactor."""
        opt, _ = optimizer
        record = opt.step()
        assert record.measurement_source == "proteingym"
        assert opt.n_measured == 45
        assert not opt.awaiting_measurements


@pytest.mark.real_data
class TestBenchLoopEndpoints:
    """The propose → export → upload → re-propose cycle over HTTP."""

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

    @pytest.fixture
    def client(self, require_real_data):
        import time

        from fastapi.testclient import TestClient

        from drucion.api.app import app

        with TestClient(app) as c:
            r = c.post("/api/experiments", json=self.PAYLOAD)
            assert r.status_code == 201, r.text
            exp_id = r.json()["id"]
            deadline = time.time() + 180
            while time.time() < deadline:
                body = c.get(f"/api/experiments/{exp_id}").json()
                if body["status"] in ("ready", "error"):
                    break
                time.sleep(0.2)
            assert body["status"] == "ready", body.get("error")
            yield c, exp_id

    def test_propose_returns_a_batch_without_measuring(self, client):
        c, exp_id = client
        before = c.get(f"/api/experiments/{exp_id}").json()["n_measured"]
        batch = c.post(f"/api/experiments/{exp_id}/propose").json()
        assert batch["round"] == 1
        assert len(batch["variants"]) == 5
        after = c.get(f"/api/experiments/{exp_id}").json()["n_measured"]
        assert after == before

    def test_propose_is_idempotent_over_http(self, client):
        c, exp_id = client
        a = c.post(f"/api/experiments/{exp_id}/propose").json()
        b = c.post(f"/api/experiments/{exp_id}/propose").json()
        assert [v["mutant"] for v in a["variants"]] == [
            v["mutant"] for v in b["variants"]
        ]

    def test_worklist_download_has_blank_measurement_column(self, client):
        c, exp_id = client
        c.post(f"/api/experiments/{exp_id}/propose")
        r = c.get(f"/api/experiments/{exp_id}/worklist.csv")
        assert r.status_code == 200
        assert "attachment;" in r.headers["content-disposition"]
        assert "# batch_token:" in r.text
        parsed = parse_measurements(r.text)
        assert parsed.measurements == {}
        assert len(parsed.skipped) == 5

    def test_prefilled_sheet_states_its_provenance(self, client):
        c, exp_id = client
        c.post(f"/api/experiments/{exp_id}/propose")
        text = c.get(f"/api/experiments/{exp_id}/worklist.csv?prefill=true").text
        assert "# PRE-FILLED:" in text
        assert "ProteinGym" in text
        assert parse_measurements(text).n_accepted == 5

    def test_full_cycle_ingests_and_reoptimizes(self, client):
        c, exp_id = client
        first = c.post(f"/api/experiments/{exp_id}/propose").json()
        sheet = c.get(f"/api/experiments/{exp_id}/worklist.csv?prefill=true").text

        r = c.post(
            f"/api/experiments/{exp_id}/measurements",
            content=sheet.encode(),
            headers={"content-type": "text/csv"},
        )
        assert r.status_code == 200, r.text
        report = r.json()
        assert report["ingested"] is True
        assert report["n_accepted"] == 5
        assert report["n_rejected"] == 0

        state = c.get(f"/api/experiments/{exp_id}").json()
        assert state["n_measured"] == 25

        second = c.post(f"/api/experiments/{exp_id}/propose").json()
        assert second["round"] == 2
        assert not set(v["mutant"] for v in second["variants"]) & set(
            v["mutant"] for v in first["variants"]
        )

    def test_uploaded_round_is_tagged_as_uploaded(self, client):
        c, exp_id = client
        c.post(f"/api/experiments/{exp_id}/propose")
        sheet = c.get(f"/api/experiments/{exp_id}/worklist.csv?prefill=true").text
        c.post(
            f"/api/experiments/{exp_id}/measurements",
            content=sheet.encode(),
            headers={"content-type": "text/csv"},
        )
        rounds = c.get(f"/api/experiments/{exp_id}/results").json()["rounds"]
        measured = [r for r in rounds if r["selected"]]
        assert measured[-1]["measurement_source"] == "uploaded"

    def test_stale_sheet_is_refused(self, client):
        c, exp_id = client
        c.post(f"/api/experiments/{exp_id}/propose")
        stale = c.get(f"/api/experiments/{exp_id}/worklist.csv?prefill=true").text
        c.post(
            f"/api/experiments/{exp_id}/measurements",
            content=stale.encode(),
            headers={"content-type": "text/csv"},
        )
        c.post(f"/api/experiments/{exp_id}/propose")  # now on round 2
        r = c.post(
            f"/api/experiments/{exp_id}/measurements",
            content=stale.encode(),
            headers={"content-type": "text/csv"},
        )
        assert r.status_code == 422
        assert "fresh worklist" in r.json()["detail"]

    def test_unusable_file_is_rejected_with_a_reason(self, client):
        c, exp_id = client
        c.post(f"/api/experiments/{exp_id}/propose")
        r = c.post(
            f"/api/experiments/{exp_id}/measurements",
            content=b"hello world\nnot a csv\n",
            headers={"content-type": "text/csv"},
        )
        assert r.status_code == 422
        assert "measured_fitness" in r.json()["detail"]

    def test_empty_upload_is_rejected(self, client):
        c, exp_id = client
        c.post(f"/api/experiments/{exp_id}/propose")
        r = c.post(
            f"/api/experiments/{exp_id}/measurements",
            content=b"",
            headers={"content-type": "text/csv"},
        )
        assert r.status_code == 422

    def test_extend_allows_going_past_the_planned_budget(self, client):
        c, exp_id = client
        for _ in range(3):
            c.post(f"/api/experiments/{exp_id}/propose")
            sheet = c.get(f"/api/experiments/{exp_id}/worklist.csv?prefill=true").text
            c.post(
                f"/api/experiments/{exp_id}/measurements",
                content=sheet.encode(),
                headers={"content-type": "text/csv"},
            )
        assert c.post(f"/api/experiments/{exp_id}/propose").status_code == 409

        r = c.post(f"/api/experiments/{exp_id}/extend?additional_rounds=2")
        assert r.status_code == 200
        assert r.json()["n_rounds"] == 5
        assert r.json()["extended_by"] == 2
        assert c.post(f"/api/experiments/{exp_id}/propose").json()["round"] == 4

    def test_extend_validates_its_argument(self, client):
        c, exp_id = client
        assert (
            c.post(f"/api/experiments/{exp_id}/extend?additional_rounds=0").status_code
            == 422
        )
