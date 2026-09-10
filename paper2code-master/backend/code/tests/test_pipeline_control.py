"""流水线调度：失败即停、超时对齐。不改各 Step 内部算法。"""

import unittest

import config
from pipeline_controller import (
    public_pipeline_status,
    run_remaining_steps,
    step_timeout_seconds,
)


class TestRunRemainingSteps(unittest.TestCase):
    def test_stops_on_error_and_does_not_retry(self):
        calls = {"n": 0}

        def run_next():
            calls["n"] += 1
            if calls["n"] == 1:
                return {"status": "success", "step": "step1"}
            return {"status": "error", "step": "step2", "error": "boom"}

        results = run_remaining_steps(run_next, sleep=lambda _s: None)

        self.assertEqual(calls["n"], 2)
        self.assertEqual(results[-1]["status"], "error")
        self.assertEqual(results[-1]["step"], "step2")

    def test_continues_after_success_until_complete(self):
        seq = [
            {"status": "success", "step": "step1"},
            {"status": "success", "step": "step2_1"},
            {"status": "complete", "step": None},
        ]
        it = iter(seq)

        results = run_remaining_steps(lambda: next(it), sleep=lambda _s: None)

        self.assertEqual([r["status"] for r in results], ["success", "success", "complete"])

    def test_continues_after_skipped(self):
        seq = [
            {"status": "skipped", "step": "step1"},
            {"status": "success", "step": "step2_1"},
            {"status": "complete", "step": None},
        ]
        it = iter(seq)

        results = run_remaining_steps(lambda: next(it), sleep=lambda _s: None)

        self.assertEqual([r["status"] for r in results], ["skipped", "success", "complete"])

    def test_busy_retries_then_proceeds(self):
        seq = [
            {"status": "busy", "step": "step1"},
            {"status": "success", "step": "step1"},
            {"status": "complete", "step": None},
        ]
        it = iter(seq)
        slept = []

        results = run_remaining_steps(lambda: next(it), sleep=slept.append)

        self.assertEqual(slept, [1])
        self.assertEqual([r["status"] for r in results], ["busy", "success", "complete"])


class TestStepTimeouts(unittest.TestCase):
    def test_llm_timeout_at_least_300s(self):
        self.assertGreaterEqual(config.TIMEOUT, 300)

    def test_step3_timeout_matches_long_llm_steps(self):
        self.assertGreaterEqual(step_timeout_seconds("step3"), 1200)

    def test_step7_keeps_long_timeout(self):
        self.assertGreaterEqual(step_timeout_seconds("step7"), 1200)

    def test_fast_steps_stay_300s(self):
        self.assertEqual(step_timeout_seconds("step1"), 300)
        self.assertEqual(step_timeout_seconds("step2_1"), 300)


class TestPublicPipelineStatus(unittest.TestCase):
    def test_error_when_idle_is_failed(self):
        status = public_pipeline_status({
            "completed": ["step1"],
            "total": 10,
            "running": False,
            "last_result": {"status": "error", "step": "step2_1", "error": "exit 1"},
        })
        self.assertEqual(status, "failed")

    def test_error_on_first_step_is_failed_not_uploaded(self):
        status = public_pipeline_status({
            "completed": [],
            "total": 10,
            "running": False,
            "last_result": {"status": "error", "step": "step1", "error": "timeout"},
        })
        self.assertEqual(status, "failed")

    def test_running_overrides_last_error(self):
        status = public_pipeline_status({
            "completed": ["step1"],
            "total": 10,
            "running": True,
            "last_result": {"status": "error", "step": "step2_1"},
        })
        self.assertEqual(status, "processing")

    def test_all_done_is_completed(self):
        status = public_pipeline_status({
            "completed": ["a"] * 10,
            "total": 10,
            "running": False,
            "last_result": {"status": "success"},
        })
        self.assertEqual(status, "completed")


if __name__ == "__main__":
    unittest.main()
