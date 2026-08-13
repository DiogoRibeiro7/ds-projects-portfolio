"""Tests for the population-basis reconciliation layer.

These cover the arithmetic the project's central claim rests on: published GDP
and published GDP per capita together determine the population inside an index,
so whether a denominator correction is owed is a checkable fact.

The regression test that matters most is
``test_ppp_is_independent_of_any_population_assumption``. Deriving the PPP from a
population — rather than from two published per-capita figures — makes the
recovery circular and silently returns whichever population was assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pt_gdp_bayes.reconciliation import (
    assess_correction_claim,
    compare_population_bases,
    derive_purchasing_power_parity,
    employment_absorbed_revision,
    employment_rate_excess_over_peers,
    implied_missing_workers,
    population_inside_index,
)


class TestPopulationInsideIndex:
    def test_recovers_the_denominator_of_a_published_per_capita_figure(self) -> None:
        # 450bn EUR over 10m people is 45,000 EUR per head.
        assert population_inside_index(gdp=450e9, gdp_pc_eur=45_000.0) == pytest.approx(1e7)

    def test_a_smaller_per_capita_figure_implies_a_larger_population(self) -> None:
        assert population_inside_index(gdp=450e9, gdp_pc_eur=40_000.0) > (
            population_inside_index(gdp=450e9, gdp_pc_eur=45_000.0)
        )

    @pytest.mark.parametrize("field", ["gdp", "gdp_pc_eur"])
    def test_rejects_non_positive_inputs(self, field: str) -> None:
        kwargs = {"gdp": 450e9, "gdp_pc_eur": 45_000.0}
        kwargs[field] = 0.0
        with pytest.raises(ValueError, match=field):
            population_inside_index(**kwargs)


class TestPurchasingPowerParity:
    def test_is_the_ratio_of_the_two_published_per_capita_figures(self) -> None:
        assert derive_purchasing_power_parity(
            gdp_pc_eur=28_390.0, gdp_pc_pps=33_825.0
        ) == pytest.approx(0.83933, rel=1e-4)

    def test_ppp_is_independent_of_any_population_assumption(self) -> None:
        """Regression test for the circular-derivation bug.

        The PPP must come from two published per-capita figures. Deriving it as
        ``gdp / population / gdp_pc_pps`` bakes in an assumption about which
        population the index used, and then "recovering" that population returns
        the assumption rather than the fact.
        """
        gdp, gdp_pc_eur, gdp_pc_pps = 306_749.6e6, 28_390.0, 33_825.0
        ppp = derive_purchasing_power_parity(gdp_pc_eur=gdp_pc_eur, gdp_pc_pps=gdp_pc_pps)

        # The correct route recovers the accounts population whichever candidate
        # population a caller might have had in mind.
        recovered = population_inside_index(gdp=gdp, gdp_pc_eur=gdp_pc_eur)
        assert recovered == pytest.approx(10_804_847, rel=1e-4)

        # The circular route returns whatever population it was handed.
        for assumed in (10_804_847.0, 11_405_627.0):
            circular_ppp = gdp / assumed / gdp_pc_pps
            circular = gdp / (circular_ppp * gdp_pc_pps)
            assert circular == pytest.approx(assumed, rel=1e-9)
            if assumed != pytest.approx(recovered, rel=1e-3):
                assert circular_ppp != pytest.approx(ppp, rel=1e-3)

    @pytest.mark.parametrize("field", ["gdp_pc_eur", "gdp_pc_pps"])
    def test_rejects_non_positive_inputs(self, field: str) -> None:
        kwargs = {"gdp_pc_eur": 28_390.0, "gdp_pc_pps": 33_825.0}
        kwargs[field] = 0.0
        with pytest.raises(ValueError, match="must be positive"):
            derive_purchasing_power_parity(**kwargs)


class TestAssessCorrectionClaim:
    def test_detects_a_correction_that_is_genuinely_owed(self, pps_inputs) -> None:
        """The Portugal 2025 case: the index is on the un-rebenchmarked basis."""
        verdict = assess_correction_claim(
            pps_inputs, year=2025, quoted_index=81.3, revised_population=11_405_627.0
        )

        assert verdict.correction_is_owed is True
        assert verdict.index_population == pytest.approx(10_804_847, rel=1e-4)
        assert verdict.ratio == pytest.approx(0.9473, rel=1e-3)
        assert verdict.denominator_only_index == pytest.approx(77.0, abs=0.1)
        assert "BELOW the revised" in verdict.summary()

    def test_detects_a_correction_that_is_already_baked_in(self, pps_inputs) -> None:
        verdict = assess_correction_claim(
            pps_inputs, year=2025, quoted_index=81.3, revised_population=10_804_847.0
        )

        assert verdict.correction_is_owed is False
        assert verdict.ratio == pytest.approx(1.0, rel=1e-4)
        assert verdict.denominator_only_index == pytest.approx(81.3, rel=1e-3)
        assert "No denominator correction is owed" in verdict.summary()

    def test_tolerance_governs_the_verdict_boundary(self, pps_inputs) -> None:
        base = 10_804_847.0
        loose = assess_correction_claim(
            pps_inputs, year=2025, quoted_index=81.3,
            revised_population=base * 1.02, tolerance=0.05,
        )
        strict = assess_correction_claim(
            pps_inputs, year=2025, quoted_index=81.3,
            revised_population=base * 1.02, tolerance=0.01,
        )

        assert loose.correction_is_owed is False
        assert strict.correction_is_owed is True

    def test_correction_scales_the_index_by_the_population_ratio(self, pps_inputs) -> None:
        verdict = assess_correction_claim(
            pps_inputs, year=2025, quoted_index=81.3, revised_population=11_405_627.0
        )
        assert verdict.denominator_only_index == pytest.approx(
            verdict.quoted_index * verdict.ratio
        )

    def test_raises_for_a_year_with_no_data(self, pps_inputs) -> None:
        with pytest.raises(KeyError, match="1999"):
            assess_correction_claim(
                pps_inputs, year=1999, quoted_index=81.3, revised_population=1e7
            )

    def test_raises_when_required_columns_are_absent(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            assess_correction_claim(
                pd.DataFrame({"year": [2025]}), year=2025,
                quoted_index=81.3, revised_population=1e7,
            )

    def test_rejects_a_non_positive_revised_population(self, pps_inputs) -> None:
        with pytest.raises(ValueError, match="revised_population"):
            assess_correction_claim(
                pps_inputs, year=2025, quoted_index=81.3, revised_population=0.0
            )


class TestComparePopulationBases:
    def test_reports_the_gap_between_the_two_official_bases(self, pps_inputs) -> None:
        out = compare_population_bases(pps_inputs)
        row = out.loc[out["year"] == 2025].iloc[0]

        assert row["gap"] == pytest.approx(11_405_627.0 - 10_804_160.0)
        assert row["gap_pct"] == pytest.approx(5.57, abs=0.05)

    def test_shows_the_index_uses_the_national_accounts_base(self, pps_inputs) -> None:
        """The claim is demonstrated from published data, not asserted."""
        out = compare_population_bases(pps_inputs)
        row = out.loc[out["year"] == 2025].iloc[0]

        assert row["implied_by_published_index"] == pytest.approx(row["na_population"], rel=1e-3)
        assert row["implied_by_published_index"] != pytest.approx(
            row["demographic_population"], rel=1e-2
        )

    def test_marks_the_official_anchor_on_its_year_only(self, pps_inputs) -> None:
        out = compare_population_bases(
            pps_inputs, revised_anchor=11_424_031.0, anchor_year=2025
        )
        assert out.loc[out["year"] == 2025, "official_revised"].iloc[0] == pytest.approx(
            11_424_031.0
        )
        assert np.isnan(out.loc[out["year"] == 2024, "official_revised"].iloc[0])

    def test_requires_anchor_year_with_anchor(self, pps_inputs) -> None:
        with pytest.raises(ValueError, match="anchor_year"):
            compare_population_bases(pps_inputs, revised_anchor=1e7)

    def test_raises_when_required_columns_are_absent(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            compare_population_bases(pd.DataFrame({"year": [2025]}))


class TestEmploymentAbsorbedRevision:
    def test_ratio_is_flat_when_employment_tracks_its_own_population_base(self) -> None:
        years = [2022, 2023, 2024, 2025]
        population = pd.DataFrame(
            {"year": years, "na_population": [10.0e6, 10.3e6, 10.6e6, 10.9e6]}
        )
        employment = pd.DataFrame(
            {"year": years, "na_employment": [4.7e6, 4.841e6, 4.982e6, 5.123e6]}
        )

        out = employment_absorbed_revision(
            na_employment=employment, na_population=population, years=(2022, 2025)
        )

        ratios = out["employment_population_ratio"]
        assert ratios.max() - ratios.min() < 1e-6

    def test_ratio_rises_when_employment_outgrows_its_population_base(self) -> None:
        """The Portugal signature: the accounts pick up workers their own
        population benchmark does not know about."""
        years = [2022, 2023, 2024, 2025]
        population = pd.DataFrame(
            {"year": years, "na_population": [10.0e6, 10.1e6, 10.2e6, 10.3e6]}
        )
        employment = pd.DataFrame(
            {"year": years, "na_employment": [4.7e6, 4.85e6, 5.0e6, 5.2e6]}
        )

        out = employment_absorbed_revision(
            na_employment=employment, na_population=population, years=(2022, 2025)
        )

        assert out["employment_population_ratio"].is_monotonic_increasing

    def test_restricts_to_the_requested_year_window(self) -> None:
        years = list(range(2018, 2026))
        population = pd.DataFrame({"year": years, "na_population": np.linspace(1e7, 1.1e7, 8)})
        employment = pd.DataFrame({"year": years, "na_employment": np.linspace(4.5e6, 5e6, 8)})

        out = employment_absorbed_revision(
            na_employment=employment, na_population=population, years=(2022, 2024)
        )

        assert out["year"].tolist() == [2022, 2023, 2024]


class TestEmploymentRateExcessOverPeers:
    RATES = pd.DataFrame(
        {
            "year": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "PT": [69.8, 68.3, 69.6, 71.4, 72.4, 72.8, 73.9],
            "EU27_2020": [68.1, 67.1, 68.3, 69.8, 70.4, 70.7, 71.0],
            "ES": [63.3, 60.9, 62.6, 64.3, 65.3, 66.1, 67.0],
            "IE": [69.4, 66.5, 69.8, 73.3, 74.0, 74.5, 74.6],
        }
    )

    def test_separates_the_common_trend_from_the_excess(self) -> None:
        out = employment_rate_excess_over_peers(
            self.RATES, country="PT", peers=["EU27_2020", "ES", "IE"],
            base_year=2019, target_year=2025,
        )

        assert out.set_index("peer").loc["EU27_2020", "excess_pp"] == pytest.approx(1.2, abs=0.01)
        assert out.set_index("peer").loc["ES", "excess_pp"] == pytest.approx(0.4, abs=0.01)
        assert out.set_index("peer").loc["IE", "excess_pp"] == pytest.approx(-1.1, abs=0.01)

    def test_the_portugal_case_shows_no_consistent_excess(self) -> None:
        """Regression test for a withdrawn finding.

        An earlier version read Portugal's employment-rate rise as evidence that
        the accounts were absorbing uncounted migrant workers. Against peers that
        also absorbed large inflows the excess is not consistently positive, so
        the rise carries no absorption signal.
        """
        out = employment_rate_excess_over_peers(
            self.RATES, country="PT", peers=["EU27_2020", "ES", "IE"],
            base_year=2019, target_year=2025,
        )
        assert out["excess_pp"].min() < 0 < out["excess_pp"].max()

    def test_country_change_is_the_same_for_every_peer(self) -> None:
        out = employment_rate_excess_over_peers(
            self.RATES, country="PT", peers=["EU27_2020", "ES"],
            base_year=2019, target_year=2025,
        )
        assert out["country_change_pp"].nunique() == 1

    def test_raises_for_an_unknown_geography(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            employment_rate_excess_over_peers(
                self.RATES, country="PT", peers=["XX"], base_year=2019, target_year=2025
            )

    def test_raises_for_a_year_with_no_data(self) -> None:
        with pytest.raises(KeyError, match="1999"):
            employment_rate_excess_over_peers(
                self.RATES, country="PT", peers=["ES"], base_year=1999, target_year=2025
            )


class TestImpliedMissingWorkers:
    def test_multiplies_the_participation_chain(self) -> None:
        assert implied_missing_workers(
            extra_residents=600_000.0, working_age_share=0.85,
            activity_rate=0.87, employment_rate=0.88,
        ) == pytest.approx(600_000 * 0.85 * 0.87 * 0.88)

    def test_no_extra_residents_means_no_missing_workers(self) -> None:
        assert implied_missing_workers(
            extra_residents=0.0, working_age_share=0.85,
            activity_rate=0.87, employment_rate=0.88,
        ) == 0.0

    @pytest.mark.parametrize(
        "field", ["working_age_share", "activity_rate", "employment_rate"]
    )
    def test_rejects_rates_outside_the_unit_interval(self, field: str) -> None:
        kwargs = {
            "extra_residents": 600_000.0, "working_age_share": 0.85,
            "activity_rate": 0.87, "employment_rate": 0.88,
        }
        kwargs[field] = 1.5
        with pytest.raises(ValueError, match=field):
            implied_missing_workers(**kwargs)

    def test_rejects_negative_extra_residents(self) -> None:
        with pytest.raises(ValueError, match="extra_residents"):
            implied_missing_workers(
                extra_residents=-1.0, working_age_share=0.85,
                activity_rate=0.87, employment_rate=0.88,
            )


class TestNonFiniteGuards:
    """NaN silently passed every ``<= 0`` guard, because ``nan <= 0`` is False.

    In :func:`assess_correction_claim` that produced ``correction_is_owed=False``
    and a confident "no correction is owed" summary — inverting the project's
    headline claim on missing data.
    """

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_assess_correction_claim_rejects_a_non_finite_population(
        self, pps_inputs, bad: float
    ) -> None:
        with pytest.raises(ValueError, match="finite"):
            assess_correction_claim(
                pps_inputs, year=2025, quoted_index=81.3, revised_population=bad
            )

    @pytest.mark.parametrize("field", ["gdp_pc_eur", "gdp_pc_pps"])
    def test_derive_ppp_rejects_non_finite_inputs(self, field: str) -> None:
        kwargs = {"gdp_pc_eur": 28_390.0, "gdp_pc_pps": 33_825.0}
        kwargs[field] = np.nan
        with pytest.raises(ValueError, match="finite"):
            derive_purchasing_power_parity(**kwargs)

    @pytest.mark.parametrize("field", ["gdp", "gdp_pc_eur"])
    def test_population_inside_index_rejects_non_finite_inputs(self, field: str) -> None:
        kwargs = {"gdp": 306_749.6e6, "gdp_pc_eur": 28_390.0}
        kwargs[field] = np.nan
        with pytest.raises(ValueError, match="finite"):
            population_inside_index(**kwargs)
