from __future__ import annotations

import unittest

from app.geomotion import GeoMotionRequest, GeoMotionResponse, simulate


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

    def test_hybrid_and_physics_modes_are_distinct(self):
        physics = simulate(request("physics"))
        hybrid = simulate(request("hybrid"))
        self.assertNotEqual(
            physics["metrics"]["mean_displacement_m"],
            hybrid["metrics"]["mean_displacement_m"],
        )
        self.assertIn("random-forest", hybrid["engine"]["model_kind"])

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


if __name__ == "__main__":
    unittest.main()
