"""审计修复的单元测试（只测确定性规则层，不调 LLM）。

覆盖：
  ① 入口 --run N 支持（CodeGenerator 输出要求 + CodeValidator 确定性检查）
  ② leaf_mapping 依赖登记（Step6 data_dependency / for_code_agent + Step7 注入）
  ③ segmented 预处理 PARTIAL 恢复 + known_best/worst 按句提取
  ④ Step4 工程参数缺口 + 跨字段冲突（对齐 Step6 gap/CONFLICT-01）
  ⑤ GPU 不可用降级 CPU 运行（warning + RUN，不再硬 SKIP）
  附：SANDBOX_TIMEOUT 调大、LLM 空响应按可重试错误处理
"""

import unittest
from unittest import mock

from step4.step4_4_gap import GapAnalyzer
from step5.step5_3_data import DatasetAnalyzer
from step5.step5_5_exp import ExperimentAnalyzer
from step6.step6_summarizer import ProjectBriefIntegrator
from step7.code_generator import CodeGenerator
from step7.code_context_builder import CodeContextBuilder
from step7.code_validator import CodeValidator
from step7.step7_main import SANDBOX_TIMEOUT, Step7Runner
from step8.experiment_scheduler import ExperimentScheduler
from step8.resource_checker import ResourceChecker


# ============================================================
# ① 入口 --run N
# ============================================================

class TestEntryRunArg(unittest.TestCase):
    def test_generator_requires_run_arg_for_entry_files(self):
        g = CodeGenerator()
        for entry in ("main.py", "run.py", "src/main.py"):
            reqs = "\n".join(g._output_requirements(entry))
            self.assertIn("--run", reqs, f"{entry} 的输出要求必须包含 --run")
            self.assertIn("experiment", reqs)
        # 非入口文件不强制
        for other in ("data/dataset.py", "config/config.yaml"):
            reqs = "\n".join(g._output_requirements(other))
            self.assertNotIn("入口文件", reqs)

    def test_supports_run_arg_detects_add_argument(self):
        with_run = (
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--run', type=int, default=0)\n"
            "p.add_argument('--config')\n"
        )
        without_run = (
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--config')\n"
        )
        self.assertTrue(CodeValidator.supports_run_arg(with_run))
        self.assertFalse(CodeValidator.supports_run_arg(without_run))

    def test_supports_run_arg_accepts_nonfirst_option(self):
        # 真实写法：-r/--run 缩写并存，--run 不在参数首位
        code = "parser.add_argument('-r', '--run', type=int)\n"
        self.assertTrue(CodeValidator.supports_run_arg(code))

    def test_entry_check_fails_without_run_when_multi_runs(self):
        v = CodeValidator(all_files=["main.py", "config/config.yaml"])
        v.set_expected_runs(60)
        errors = []
        ok = v._check_entry_run_arg(
            "main.py", "def main():\n    pass\n", errors)
        self.assertFalse(ok)
        self.assertTrue(errors and "--run" in errors[0])

    def test_entry_check_passes_with_run_arg(self):
        v = CodeValidator(all_files=["main.py"])
        v.set_expected_runs(60)
        code = (
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--run', type=int)\n"
        )
        errors = []
        self.assertTrue(v._check_entry_run_arg("main.py", code, errors))
        self.assertEqual(errors, [])

    def test_single_run_or_no_run_does_not_require(self):
        v = CodeValidator(all_files=["main.py"])
        errors = []
        v.set_expected_runs(1)
        self.assertTrue(
            v._check_entry_run_arg("main.py", "print('hi')\n", errors))
        v.set_expected_runs(0)
        self.assertTrue(
            v._check_entry_run_arg("main.py", "print('hi')\n", errors))

    def test_run_count_falls_back_to_config_runs(self):
        v = CodeValidator(all_files=["main.py"])
        v.set_config({"experiment": {"runs": [{"id": "r1"}, {"id": "r2"}]}})
        errors = []
        self.assertFalse(
            v._check_entry_run_arg("main.py", "print('hi')\n", errors))
        self.assertTrue(errors)

    def test_non_entry_file_not_checked(self):
        v = CodeValidator(all_files=["data/dataset.py"])
        v.set_expected_runs(60)
        errors = []
        self.assertTrue(
            v._check_entry_run_arg("data/dataset.py", "x = 1\n", errors))

    def test_validate_wires_entry_check(self):
        v = CodeValidator(all_files=["main.py"])
        v.set_expected_runs(60)
        result = v.validate("main.py", "def main():\n    pass\n", {})
        self.assertEqual(result["status"], "failed")
        self.assertIn("entry_run_arg", result["checks"])
        self.assertTrue(any("--run" in e for e in result["errors"]))

    def test_entry_supports_run_arg_delegates_to_validator(self):
        generated = {"main.py": "import argparse\n"
                                "p = argparse.ArgumentParser()\n"
                                "p.add_argument('--run', type=int)\n"}
        self.assertTrue(
            Step7Runner._entry_supports_run_arg("main.py", generated))
        self.assertFalse(Step7Runner._entry_supports_run_arg(
            "main.py", {"main.py": "print('no args')\n"}))

    def test_sandbox_timeout_raised(self):
        self.assertGreaterEqual(SANDBOX_TIMEOUT, 120)


# ============================================================
# ② leaf_mapping 依赖登记
# ============================================================

def _leaf_integrator():
    integ = ProjectBriefIntegrator.__new__(ProjectBriefIntegrator)
    integ.paper_name = "test"
    integ.step4_yamls = {"functional_requirements": {"requirements": []}}
    integ.step5_yamls = {
        "dataset": {
            "data_source": "external_dataset",
            "acquisition": "local_path",
            "split_strategy": {"method": "leaf_based_split"},
        },
        "training_config": {},
    }
    return integ


class TestLeafMapping(unittest.TestCase):
    def test_leaf_mapping_registered_on_leaf_split(self):
        dd = _leaf_integrator()._build_data_dependency()
        self.assertIn("leaf_mapping", dd)
        leaf = dd["leaf_mapping"]
        self.assertTrue(leaf["required"])
        self.assertEqual(leaf["file"], "data/leaf_mappings.csv")
        self.assertIn("leaf_constraint_effective=false", leaf["on_missing"])
        self.assertIn("不得静默回退", leaf["on_missing"])

    def test_no_leaf_mapping_on_random_split(self):
        integ = _leaf_integrator()
        integ.step5_yamls["dataset"]["split_strategy"] = {"method": "random_split"}
        self.assertNotIn("leaf_mapping", integ._build_data_dependency())

    def test_for_code_agent_carries_leaf_constraint(self):
        notes = "\n".join(_leaf_integrator()._for_code_agent_notes())
        self.assertIn("leaf_mapping", notes)
        self.assertIn("leaf_constraint_effective=false", notes)
        self.assertIn("严禁静默回退", notes)

    def test_context_builder_merges_leaf_into_split_strategy(self):
        brief = {
            "data_dependency": {"leaf_mapping": {
                "required": True,
                "file": "data/leaf_mappings.csv",
                "constraint": "同一片叶子不得跨集",
                "on_missing": "必须告警并记录 leaf_constraint_effective=false",
            }},
            "code_generation_plan": {"per_file_specs": []},
        }
        builder = CodeContextBuilder.__new__(CodeContextBuilder)
        builder.brief = brief
        ss = builder._split_strategy_with_leaf(
            {"split_strategy": {"method": "leaf_based_split"}})
        self.assertEqual(ss["leaf_mapping_file"], "data/leaf_mappings.csv")
        self.assertIn("不得跨集", ss["leaf_mapping_constraint"])
        self.assertIn("leaf_constraint_effective=false", ss["leaf_mapping_on_missing"])

    def test_context_builder_no_leaf_without_registration(self):
        builder = CodeContextBuilder.__new__(CodeContextBuilder)
        builder.brief = {"data_dependency": {}}
        ss = builder._split_strategy_with_leaf({"split_strategy": {}})
        self.assertNotIn("leaf_mapping_file", ss)

    def test_inject_data_dependency_writes_leaf_keys(self):
        runner = Step7Runner("leaf_inject_test")
        runner._write_code = lambda f, c: None  # 不落盘
        generated = {
            "config/config.yaml": "data_source: external_dataset\n"
                                  "acquisition: local_path\n"
                                  "split_strategy:\n  method: leaf_based_split\n",
        }
        brief = {"data_dependency": {
            "data_source": "external_dataset",
            "acquisition": "local_path",
            "leaf_mapping": {
                "required": True,
                "file": "data/leaf_mappings.csv",
                "constraint": "同一片叶子的多张图片必须保持在同一个集合中",
                "on_missing": "必须 logger.warning 并记录 leaf_constraint_effective=false",
            },
        }}
        validator = CodeValidator(all_files=["config/config.yaml"])
        import yaml
        runner._inject_data_dependency(generated, brief, validator)
        cfg = yaml.safe_load(generated["config/config.yaml"])
        self.assertEqual(
            cfg["split_strategy"]["leaf_mapping_file"], "data/leaf_mappings.csv")
        self.assertIn("同一片叶子", cfg["split_strategy"]["leaf_mapping_constraint"])
        self.assertIn("leaf_constraint_effective=false",
                      cfg["split_strategy"]["leaf_mapping_on_missing"])


# ============================================================
# ③ segmented PARTIAL + known_extremes 按句提取
# ============================================================

class TestSegmentationRecovery(unittest.TestCase):
    def test_described_method_gives_partial(self):
        ds = {"preprocessing": (
            "Segmented version: images are converted to HSB color space; "
            "hue and saturation components are analyzed to generate a leaf mask; "
            "color casts are fixed before segmentation."
        )}
        seg = DatasetAnalyzer._extract_segmentation_algorithm(ds)
        self.assertTrue(seg["implementation_status"].startswith("PARTIAL"))
        self.assertIn("HSB", seg["color_spaces"])
        self.assertIn("hue", seg["components"])
        self.assertTrue(seg["fixed_color_casts"])
        self.assertIn("NOT_FOUND", seg["parameters"])

    def test_undescribed_gives_not_found(self):
        ds = {"preprocessing": "Segmented images are used."}
        seg = DatasetAnalyzer._extract_segmentation_algorithm(ds)
        self.assertTrue(seg["implementation_status"].startswith("NOT_FOUND"))
        self.assertEqual(seg["color_spaces"], [])

    def test_empty_preprocessing_not_found(self):
        seg = DatasetAnalyzer._extract_segmentation_algorithm({})
        self.assertTrue(seg["implementation_status"].startswith("NOT_FOUND"))

    def test_lab_color_space_detected(self):
        ds = {"preprocessing": "mask via L*a*b* lightness channel segmentation"}
        seg = DatasetAnalyzer._extract_segmentation_algorithm(ds)
        self.assertIn("L*a*b*", seg["color_spaces"])
        self.assertIn("lightness", seg["components"])


class TestKnownExtremes(unittest.TestCase):
    COMPARISON = (
        "Best performance: GoogLeNet::TransferLearning::Color::80-20 with "
        "mean F1 score 0.9934 and overall accuracy 99.35%. "
        "Worst performance: AlexNet::TrainingFromScratch::GrayScale::80-20 "
        "with mean F1 score 0.8524 and overall accuracy 85.53%. "
        "On real-world images (121 images from trusted online sources), "
        "GoogLeNet:Segmented:TransferLearning:80-20 achieves 31.40% accuracy, "
        "47.93% when crop information is provided, 52.89% top-5 accuracy."
    )

    def test_best_sentence_only(self):
        out = ExperimentAnalyzer._extract_known_extremes(self.COMPARISON)
        self.assertIn("99.35%", out["known_best"])
        self.assertIn("GoogLeNet::TransferLearning::Color::80-20",
                      out["known_best"])
        # 不得把真实场景句的括号说明（121 images）混记进 Best
        self.assertNotIn("121", out["known_best"])

    def test_worst_sentence_only(self):
        out = ExperimentAnalyzer._extract_known_extremes(self.COMPARISON)
        self.assertIn("85.53%", out["known_worst"])
        self.assertIn("AlexNet::TrainingFromScratch::GrayScale::80-20",
                      out["known_worst"])

    def test_real_world_separate(self):
        out = ExperimentAnalyzer._extract_known_extremes(self.COMPARISON)
        self.assertIn("31.40%", out["known_real_world"])
        self.assertIn("real-world", out["known_real_world"])
        # 主值取句首 31.40%，其余作为附注
        self.assertIn("47.93", out["known_real_world"])

    def test_hyphenated_config_not_truncated(self):
        out = ExperimentAnalyzer._extract_known_extremes(self.COMPARISON)
        self.assertIn("80-20", out["known_best"])

    def test_no_comparison_returns_not_found(self):
        out = ExperimentAnalyzer._extract_known_extremes("NOT_FOUND")
        self.assertEqual(out["known_best"], "NOT_FOUND")
        self.assertEqual(out["known_worst"], "NOT_FOUND")
        self.assertEqual(out["known_real_world"], "NOT_FOUND")

    def test_matrix_emits_known_keys(self):
        ana = ExperimentAnalyzer.__new__(ExperimentAnalyzer)
        ana.paper_name = "test"
        ana.key_info = {
            "methodology": {"model_architecture": {"alexnet": "AlexNet"}},
            "implementation_details": {"optimizer": "SGD"},
            "dataset": {"data_type": "image", "train_test_split": "80-20"},
            "experiment": {"comparison_results": self.COMPARISON},
        }
        matrix = ana._build_experiment_matrix()
        self.assertIn("known_best", matrix)
        self.assertIn("known_worst", matrix)
        self.assertIn("known_real_world", matrix)
        self.assertIn("99.35%", matrix["known_best"])


# ============================================================
# ④ Step4 工程参数缺口 + 跨字段冲突
# ============================================================

def _gap_analyzer(key_info):
    ga = GapAnalyzer.__new__(GapAnalyzer)
    ga.paper_name = "test"
    ga.key_info = key_info
    ga._counters = {GapAnalyzer.GAP_PREFIX: 0}
    return ga


class TestEngineeringParamGaps(unittest.TestCase):
    def _arch(self):
        return {"pretrained_model": "ImageNet pretrained weights"}

    def test_missing_params_flagged(self):
        ga = _gap_analyzer({
            "implementation_details": {
                "optimizer": "SGD", "learning_rate": "0.005", "epoch": "30"},
            "dataset": {"train_test_split": "80-20 train/test"},
        })
        gaps = ga._check_engineering_parameter_gaps(ga.key_info, self._arch())
        paths = [g["path"] for g in gaps]
        self.assertIn("training.early_stopping", paths)
        self.assertIn("checkpoint.save_frequency_epochs", paths)
        self.assertIn("inference.batch_size", paths)
        self.assertIn("training.freeze_backbone", paths)
        for g in gaps:
            self.assertEqual(g["severity"], "low")
            self.assertIn("config.yaml", g["suggested"])

    def test_mentioned_params_not_flagged(self):
        ga = _gap_analyzer({
            "implementation_details": {
                "optimizer": "SGD",
                "training_strategy": (
                    "Early stopping with patience=5; checkpoint saved every "
                    "epoch; inference batch size 32; backbone frozen during "
                    "transfer learning."),
            },
        })
        gaps = ga._check_engineering_parameter_gaps(ga.key_info, self._arch())
        self.assertEqual(gaps, [])

    def test_freeze_backbone_skipped_without_pretrained(self):
        ga = _gap_analyzer({"implementation_details": {"optimizer": "SGD"}})
        gaps = ga._check_engineering_parameter_gaps(ga.key_info, {})
        paths = [g["path"] for g in gaps]
        self.assertNotIn("training.freeze_backbone", paths)
        self.assertIn("training.early_stopping", paths)


class TestCrossFieldConflicts(unittest.TestCase):
    def test_train_test_only_with_training_conflicts(self):
        # 对齐 Step6 CONFLICT-01：train/test 二路 + 训练需模型选择
        ga = _gap_analyzer({
            "dataset": {"train_test_split": "80-20 train-test split"},
            "implementation_details": {"optimizer": "SGD", "epoch": "30"},
        })
        gaps = ga._check_cross_field_conflicts(ga.key_info, is_training=True)
        self.assertEqual(len(gaps), 1)
        g = gaps[0]
        self.assertEqual(g["path"], "dataset.train_test_split")
        self.assertIn("跨字段冲突", g["question"])
        self.assertIn("数据泄露", g["question"])
        self.assertEqual(g["severity"], "medium")
        self.assertTrue(any("二次划分" in a for a in g["alternatives"]))

    def test_val_split_no_conflict(self):
        ga = _gap_analyzer({
            "dataset": {"train_test_split": "70/15/15 train/val/test"},
        })
        self.assertEqual(
            ga._check_cross_field_conflicts(ga.key_info, is_training=True), [])

    def test_non_training_paper_no_conflict(self):
        ga = _gap_analyzer({
            "dataset": {"train_test_split": "80-20 train/test"},
        })
        self.assertEqual(
            ga._check_cross_field_conflicts(ga.key_info, is_training=False), [])

    def test_chinese_val_word_recognized(self):
        ga = _gap_analyzer({
            "dataset": {"train_test_split": "按 7:1:2 划分训练集/验证集/测试集"},
        })
        self.assertEqual(
            ga._check_cross_field_conflicts(ga.key_info, is_training=True), [])


# ============================================================
# ⑤ GPU 降级 CPU
# ============================================================

class TestGpuRelaxation(unittest.TestCase):
    def _checker(self, gpu_available=False):
        rc = ResourceChecker.__new__(ResourceChecker)
        rc.environment = {
            "gpu_available": gpu_available,
            "gpu_memory_gb": 24.0 if gpu_available else None,
        }
        return rc

    def test_gpu_unavailable_degrades_to_cpu(self):
        res = self._checker().check_gpu({"gpu_required": True})
        self.assertTrue(res["sufficient"])
        self.assertIsNone(res["reason"])
        self.assertIn("gpu_unavailable_cpu_fallback", res["warnings"])
        self.assertIn("CPU", res["detail"])

    def test_gpu_not_required_still_ok(self):
        res = self._checker().check_gpu({"gpu_required": False})
        self.assertTrue(res["sufficient"])
        self.assertEqual(res["warnings"], [])

    def test_scheduler_runs_with_gpu_warning(self):
        checker = self._checker(gpu_available=False)
        checker.check_all = lambda exp: {
            "gpu": checker.check_gpu(exp),
        }
        scheduler = ExperimentScheduler()
        exps = [{
            "id": "run_001", "run_index": 0, "model": "Alexnet",
            "mode": "scratch", "dataset_version": "color",
            "train_ratio": 0.8, "gpu_required": True,
        }]
        plan = scheduler.build_plan(exps, checker, spec=None, config_path=None,
                                    project=None)
        e = plan["experiments"][0]
        self.assertEqual(e["decision"], "RUN")
        self.assertEqual(e["skip_reasons"], [])
        self.assertTrue(any(
            w["reason"] == "gpu_unavailable_cpu_fallback" for w in e["warnings"]))


# ============================================================
# LLM 空响应按可重试错误处理
# ============================================================

class TestEmptyLlmResponse(unittest.TestCase):
    def _post(self, content):
        resp = mock.MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]}
        return resp

    def test_empty_response_raises_and_retries(self):
        import time
        import llm_manager
        with mock.patch.object(llm_manager.requests, "post",
                               return_value=self._post("")) as post, \
             mock.patch.object(llm_manager.time, "sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                llm_manager._http_post_with_retry(
                    {}, {"messages": []}, "http://x", timeout=5,
                    max_retries=2, model_key="test-empty")
            self.assertIn("empty", str(ctx.exception).lower())
            self.assertEqual(post.call_count, 2)  # 重试而非直接返回空

    def test_nonempty_response_returned(self):
        import llm_manager
        with mock.patch.object(llm_manager.requests, "post",
                               return_value=self._post("hello")):
            out = llm_manager._http_post_with_retry(
                {}, {"messages": []}, "http://x", timeout=5,
                max_retries=2, model_key="test-nonempty")
            self.assertEqual(out, "hello")


if __name__ == "__main__":
    unittest.main()
