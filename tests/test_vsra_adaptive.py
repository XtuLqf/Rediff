import tempfile
import unittest
from pathlib import Path

from scripts.sweep_rel_con_weight import MODALITIES, parse_log
from vsra_adaptive import VSRAAdaptiveGate, compute_reliability


class ReliabilityTests(unittest.TestCase):
    def test_perfect_agreement_has_full_reliability(self):
        scores = compute_reliability(d_cs=0.0, d_vs=0.2, d_vc=0.2)
        self.assertAlmostEqual(scores["reliability"], 1.0)

    def test_conflict_and_disadvantage_reduce_reliability(self):
        baseline = compute_reliability(d_cs=0.0, d_vs=0.2, d_vc=0.2)["reliability"]
        conflict = compute_reliability(d_cs=0.2, d_vs=0.2, d_vc=0.2)["reliability"]
        disadvantage = compute_reliability(d_cs=0.0, d_vs=0.2, d_vc=0.8)["reliability"]
        self.assertLess(conflict, baseline)
        self.assertLess(disadvantage, baseline)

    def test_score_is_finite_and_bounded(self):
        for values in ((0, 0, 0), (1e30, 1e-30, 1e30), (-1, -2, -3)):
            score = compute_reliability(*values)["reliability"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class AdaptiveGateTests(unittest.TestCase):
    def make_gate(self):
        return VSRAAdaptiveGate(
            max_weight=1.0,
            ema_decay=0.5,
            temperature=1.0,
            warmup_ratio=0.5,
            total_steps=4,
        )

    def test_warmup_and_weight_bounds(self):
        gate = self.make_gate()
        first = gate.update(0.0, 0.2, 0.2)
        second = gate.update(0.0, 0.2, 0.2)
        self.assertAlmostEqual(first.weight, 0.5)
        self.assertAlmostEqual(second.weight, 1.0)
        self.assertLessEqual(second.weight, gate.max_weight)

    def test_ema_smooths_reliability(self):
        gate = self.make_gate()
        first = gate.update(0.0, 0.2, 0.2)
        second = gate.update(1.0, 0.0, 1.0)
        self.assertLess(second.raw_reliability, first.raw_reliability)
        self.assertGreater(second.ema_reliability, second.raw_reliability)

    def test_state_round_trip_continues_trajectory(self):
        original = self.make_gate()
        original.update(0.0, 0.2, 0.2)
        state = original.state_dict()

        restored = self.make_gate()
        restored.load_state_dict(state)
        expected = original.update(0.4, 0.2, 0.6)
        actual = restored.update(0.4, 0.2, 0.6)
        self.assertEqual(restored.step, original.step)
        self.assertAlmostEqual(actual.weight, expected.weight)
        self.assertAlmostEqual(actual.ema_reliability, expected.ema_reliability)


class SweepLogParserTests(unittest.TestCase):
    def test_extracts_all_modalities(self):
        lines = []
        for index, modality in enumerate(MODALITIES, start=1):
            lines.append(f"best GZSL ({modality}): U: 0.{index}000, S: 0.5000, H: 0.3000")
            lines.append(f"best ZSL ({modality}): 0.4000")
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "train.log"
            log_path.write_text("\n".join(lines), encoding="utf-8")
            metrics = parse_log(log_path)
        self.assertEqual(set(metrics), set(MODALITIES))
        self.assertEqual(metrics["VCS"]["zsl_acc"], "0.4000")


if __name__ == "__main__":
    unittest.main()
