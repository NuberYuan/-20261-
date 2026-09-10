"""Step3 三路调度：文本 / 表格 / 图片并行，合并结果与 skip 语义保持不变。"""

import time
import unittest

from step3.step3_extract_information import execute_step3


def _slow(name, events, duration, payload):
    def fn(paper):
        events[name]["start"] = time.perf_counter()
        time.sleep(duration)
        events[name]["end"] = time.perf_counter()
        events[name]["paper"] = paper
        return payload

    return fn


class TestStep3ParallelDispatch(unittest.TestCase):
    def test_text_tables_figures_run_concurrently(self):
        events = {"text": {}, "tables": {}, "figures": {}}
        paper = {"tables": [{"id": 1}], "images": [{"path": "a.png"}]}

        result = execute_step3(
            paper,
            skip_tables=False,
            skip_figures=False,
            extract_text=_slow(
                "text", events, 0.25, {"title": "t", "figures": [{"id": 1}]}
            ),
            analyze_tables=_slow(
                "tables", events, 0.25, {"tables_analyzed": 1, "table_results": []}
            ),
            analyze_figures=_slow("figures", events, 0.25, [{"id": 1, "type": "other"}]),
        )

        starts = [events[k]["start"] for k in ("text", "tables", "figures")]
        ends = [events[k]["end"] for k in ("text", "tables", "figures")]
        self.assertLess(
            max(starts),
            min(ends),
            "三路应重叠执行：最晚启动须早于最早结束",
        )
        self.assertEqual(result["title"], "t")
        self.assertEqual(result["table_analysis"]["tables_analyzed"], 1)
        self.assertEqual(result["image_analysis"][0]["id"], 1)
        self.assertEqual(result["paper_figures"], [{"id": 1}])

    def test_skip_tables_does_not_call_analyzer_or_set_key(self):
        called = {"tables": False}

        def extract_text(paper):
            return {"figures": []}

        def analyze_tables(paper):
            called["tables"] = True
            return {"tables_analyzed": 1, "table_results": []}

        def analyze_figures(paper):
            return []

        result = execute_step3(
            {"tables": [{"id": 1}]},
            skip_tables=True,
            skip_figures=False,
            extract_text=extract_text,
            analyze_tables=analyze_tables,
            analyze_figures=analyze_figures,
        )

        self.assertFalse(called["tables"])
        self.assertNotIn("table_analysis", result)
        self.assertIn("image_analysis", result)
        self.assertEqual(result["paper_figures"], [])

    def test_skip_figures_does_not_call_analyzer_or_set_key(self):
        called = {"figures": False}

        def extract_text(paper):
            return {"figures": [{"id": 9}]}

        def analyze_tables(paper):
            return {"tables_analyzed": 0, "table_results": []}

        def analyze_figures(paper):
            called["figures"] = True
            return [{"id": 1}]

        result = execute_step3(
            {"images": [{"path": "a.png"}]},
            skip_tables=False,
            skip_figures=True,
            extract_text=extract_text,
            analyze_tables=analyze_tables,
            analyze_figures=analyze_figures,
        )

        self.assertFalse(called["figures"])
        self.assertNotIn("image_analysis", result)
        self.assertIn("table_analysis", result)
        self.assertEqual(result["paper_figures"], [{"id": 9}])

    def test_missing_figure_prompt_omits_image_analysis(self):
        def extract_text(paper):
            return {"figures": []}

        def analyze_tables(paper):
            return {"tables_analyzed": 0, "table_results": []}

        def analyze_figures(paper):
            return None

        result = execute_step3(
            {},
            skip_tables=False,
            skip_figures=False,
            extract_text=extract_text,
            analyze_tables=analyze_tables,
            analyze_figures=analyze_figures,
        )

        self.assertNotIn("image_analysis", result)
        self.assertIn("table_analysis", result)


if __name__ == "__main__":
    unittest.main()
