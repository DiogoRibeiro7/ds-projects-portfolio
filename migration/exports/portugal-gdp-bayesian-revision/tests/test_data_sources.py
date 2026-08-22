"""Tests for loading, caching and patching data.

Network calls are never made here: the caching layer is exercised with a stub
loader, and the parsing helpers with fixtures written to a temporary directory.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pt_gdp_bayes.data_sources import (
    cached_frame,
    default_cache_dir,
    get_known_value,
    load_known_observations,
    patch_manual_observation,
)


class TestCachedFrame:
    def test_calls_the_loader_once_and_reuses_the_cache(self, tmp_path) -> None:
        calls = []

        def loader() -> pd.DataFrame:
            calls.append(1)
            return pd.DataFrame({"year": [2024], "value": [1.0]})

        first = cached_frame(loader, cache_dir=tmp_path, key="demo")
        second = cached_frame(loader, cache_dir=tmp_path, key="demo")

        assert len(calls) == 1
        pd.testing.assert_frame_equal(first, second)
        assert (tmp_path / "demo.csv").exists()

    def test_refresh_forces_a_reload(self, tmp_path) -> None:
        calls = []

        def loader() -> pd.DataFrame:
            calls.append(1)
            return pd.DataFrame({"year": [2024], "value": [float(len(calls))]})

        cached_frame(loader, cache_dir=tmp_path, key="demo")
        refreshed = cached_frame(loader, cache_dir=tmp_path, key="demo", refresh=True)

        assert len(calls) == 2
        assert refreshed["value"].iloc[0] == 2.0

    def test_distinct_keys_do_not_share_a_cache_file(self, tmp_path) -> None:
        cached_frame(lambda: pd.DataFrame({"year": [1], "a": [1]}), cache_dir=tmp_path, key="one")
        cached_frame(lambda: pd.DataFrame({"year": [1], "b": [2]}), cache_dir=tmp_path, key="two")

        assert {"a"} == set(pd.read_csv(tmp_path / "one.csv").columns) - {"year"}
        assert {"b"} == set(pd.read_csv(tmp_path / "two.csv").columns) - {"year"}

    def test_a_none_cache_dir_disables_caching(self, tmp_path) -> None:
        calls = []

        def loader() -> pd.DataFrame:
            calls.append(1)
            return pd.DataFrame({"year": [2024], "value": [1.0]})

        cached_frame(loader, cache_dir=None, key="demo")
        cached_frame(loader, cache_dir=None, key="demo")

        assert len(calls) == 2
        assert list(tmp_path.iterdir()) == []

    def test_creates_a_missing_cache_directory(self, tmp_path) -> None:
        nested = tmp_path / "a" / "b"
        cached_frame(lambda: pd.DataFrame({"year": [1]}), cache_dir=nested, key="demo")
        assert (nested / "demo.csv").exists()


def test_default_cache_dir_points_at_data_raw(tmp_path) -> None:
    assert default_cache_dir(tmp_path) == tmp_path / "data" / "raw"


class TestKnownObservations:
    HEADER = "variable,country,year,value,unit,source,notes\n"

    def _write(self, tmp_path, body: str):
        path = tmp_path / "known.csv"
        path.write_text(self.HEADER + body, encoding="utf-8")
        return path

    def test_loads_and_coerces_types(self, tmp_path) -> None:
        path = self._write(
            tmp_path, "resident_population,Portugal,2025,11424031,persons,INE,note\n"
        )
        frame = load_known_observations(path)

        assert frame["year"].dtype.kind == "i"
        assert frame["value"].iloc[0] == pytest.approx(11_424_031.0)

    def test_rejects_a_file_missing_required_columns(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("variable,year\nx,2025\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Missing required columns"):
            load_known_observations(path)

    def test_get_known_value_returns_the_single_match(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            "resident_population,Portugal,2025,11424031,persons,INE,n\n"
            "real_gdp_growth,Portugal,2025,0.019,decimal,INE,n\n",
        )
        observations = load_known_observations(path)

        assert get_known_value(observations, "real_gdp_growth", 2025) == pytest.approx(0.019)

    def test_get_known_value_raises_for_an_unknown_variable(self, tmp_path) -> None:
        path = self._write(tmp_path, "resident_population,Portugal,2025,1,persons,INE,n\n")
        observations = load_known_observations(path)

        with pytest.raises(KeyError, match="missing_variable"):
            get_known_value(observations, "missing_variable", 2025)

    def test_get_known_value_raises_on_duplicates(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            "dup,Portugal,2025,1,persons,INE,n\ndup,Portugal,2025,2,persons,INE,n\n",
        )
        observations = load_known_observations(path)

        with pytest.raises(ValueError, match="Multiple observations"):
            get_known_value(observations, "dup", 2025)


class TestPatchManualObservation:
    def test_updates_an_existing_year_in_place(self) -> None:
        frame = pd.DataFrame({"year": [2024, 2025], "population": [10.0, 11.0]})
        out = patch_manual_observation(frame, year=2025, column="population", value=12.0)

        assert out.loc[out["year"] == 2025, "population"].iloc[0] == 12.0
        assert out.loc[out["year"] == 2024, "population"].iloc[0] == 10.0
        assert len(out) == 2

    def test_appends_a_missing_year(self) -> None:
        frame = pd.DataFrame({"year": [2024], "population": [10.0]})
        out = patch_manual_observation(frame, year=2025, column="population", value=12.0)

        assert out["year"].tolist() == [2024, 2025]
        assert out.loc[out["year"] == 2025, "population"].iloc[0] == 12.0

    def test_creates_the_column_when_absent(self) -> None:
        frame = pd.DataFrame({"year": [2024, 2025]})
        out = patch_manual_observation(frame, year=2025, column="population", value=12.0)

        assert "population" in out.columns
        assert out.loc[out["year"] == 2025, "population"].iloc[0] == 12.0

    def test_records_the_provenance_note(self) -> None:
        frame = pd.DataFrame({"year": [2025], "population": [10.0]})
        out = patch_manual_observation(
            frame, year=2025, column="population", value=11.0, note="INE revision"
        )

        assert out.loc[out["year"] == 2025, "manual_patch_notes"].iloc[0] == "INE revision"

    def test_requires_a_year_column(self) -> None:
        with pytest.raises(ValueError, match="'year' column"):
            patch_manual_observation(
                pd.DataFrame({"population": [1.0]}), year=2025, column="population", value=1.0
            )
