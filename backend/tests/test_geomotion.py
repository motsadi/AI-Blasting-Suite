from __future__ import annotations

import unittest

from app.geomotion import GeoMotionRequest, GeoMotionResponse, simulate
from app.geomotion.explosives import linear_charge_kg_m
from app.geomotion.physics.scheduler import build_event_queue


def request(mode: str = "hybrid") -> GeoMotionRequest:
    holes = []
    for row in range(4):
        for column in range(5):
            depth = 15.5 + 0.05 * row
            holes.append(
                {
                    "id": f"{chr(65 + row)}{column + 1}",
                    "x": -5400.0 + column * 7.0,
                    "y": 4580.0 + row * 6.0,
                    "z": 664.0 + depth,
                    "depth": depth,
                    "charge": 625.0,
                    "delay_ms": 8000 + (row * 5 + column) * 8,
                }
            )
    return GeoMotionRequest(project_name="Test", seed=42, mode=mode, holes=holes)


class GeoMotionEngineTests(unittest.TestCase):
    def test_seeded_run_is_deterministic_and_mass_conserving(self):
        first = simulate(request())
        second = simulate(request())
        GeoMotionResponse.model_validate(first)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["blocks"][:5], second["blocks"][:5])
        self.assertEqual(first["metrics"]["mass_balance_error_percent"], 0.0)
        self.assertGreater(first["metrics"]["total_tonnes"], 0)
        self.assertEqual(first["metrics"]["voxel_size_m"], 1.0)
        self.assertTrue(first["remap"]["mass_preserved"])
        self.assertTrue(first["events"])

    def test_hybrid_and_physics_modes_are_distinct(self):
        physics = simulate(request("physics"))
        hybrid = simulate(request("hybrid"))
        self.assertNotEqual(
            physics["metrics"]["mean_displacement_m"],
            hybrid["metrics"]["mean_displacement_m"],
        )
        self.assertIn("dynamic relief", hybrid["engine"]["model_kind"])

    def test_metrics_are_bounded(self):
        result = simulate(request())
        for field in (
            "ore_recovery_percent",
            "ore_loss_percent",
            "dilution_percent",
            "carat_recovery_percent",
        ):
            self.assertGreaterEqual(result["metrics"][field], 0)
            self.assertLessEqual(result["metrics"][field], 100)
        self.assertAlmostEqual(
            result["metrics"]["ore_recovery_percent"]
            + result["metrics"]["ore_loss_percent"],
            100.0,
            places=1,
        )

    def test_validation_flags_duplicates_and_overlap(self):
        payload = request("physics").model_dump()
        payload["holes"][1]["id"] = payload["holes"][0]["id"]
        payload["holes"][1]["x"] = payload["holes"][0]["x"] + 0.05
        payload["holes"][1]["y"] = payload["holes"][0]["y"]
        result = simulate(GeoMotionRequest(**payload))
        self.assertEqual(result["validation"]["status"], "review")
        self.assertTrue(result["validation"]["duplicate_ids"])
        self.assertTrue(result["validation"]["near_overlap_pairs"])

    def test_delays_are_normalized_and_scatter_is_reproducible(self):
        first = build_event_queue(request(), realization=3)
        second = build_event_queue(request(), realization=3)
        self.assertEqual(first, second)
        self.assertEqual(min(event.nominal_time_ms for event in first), 0.0)
        self.assertTrue(any(abs(event.timing_error_ms) > 0 for event in first))

    def test_s135b_linear_charge_matches_site_values(self):
        self.assertAlmostEqual(linear_charge_kg_m(127), 16.0, delta=0.35)
        self.assertAlmostEqual(linear_charge_kg_m(165), 27.0, delta=0.4)
        self.assertAlmostEqual(linear_charge_kg_m(250), 61.4, delta=0.3)


if __name__ == "__main__":
    unittest.main()
