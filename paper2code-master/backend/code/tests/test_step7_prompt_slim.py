"""Step7 prompt 瘦身（接口摘要）与模型分层路由的单元测试。"""

import unittest

from step7.code_validator import CodeValidator
from step7.code_context_builder import CodeContextBuilder
from step7.code_generator import task_type_for_file, CodeGenerator
from step7.code_repairer import CodeRepairer

SAMPLE_PY = '''"""模型定义模块。"""

import torch
import torch.nn as nn

MAX_DEPTH = 4
_threshold = 0.5


class Alexnet(nn.Module):
    """AlexNet 分类网络。"""

    dropout: float = 0.5

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.secret_impl_marker = "do-not-leak"
        self.layers = nn.Sequential(nn.Conv2d(in_channels, 64, 3))

    def forward(self, x, *, training: bool = False):
        # 实现细节不应进入摘要
        return self.layers(x)


def build_model(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """构造模型实例的工厂函数。"""
    hidden = "do-not-leak-either"
    return Alexnet(num_classes, pretrained)
'''

SAMPLE_YAML = '''project:
  name: demo
  framework: pytorch
training:
  epochs: 10
  optimizer: sgd
  learning_rate: 0.001
  layers: [64, 128, 256, 512, 1024, 2048]
data:
  root: ./data
  versions:
    color:
      status: resolved
'''


class TestInterfaceDigest(unittest.TestCase):
    def test_class_and_method_signatures_present(self):
        digest = CodeValidator.interface_digest(SAMPLE_PY)
        self.assertIn("class Alexnet(nn.Module):", digest)
        self.assertIn("def __init__(self, num_classes: int=10, in_channels: int=3):", digest)
        self.assertIn("def forward(self, x, *, training: bool=False):", digest)
        self.assertIn("def build_model(num_classes: int=10, pretrained: bool=False) -> nn.Module:", digest)

    def test_docstring_first_line_present(self):
        digest = CodeValidator.interface_digest(SAMPLE_PY)
        self.assertIn("AlexNet 分类网络", digest)
        self.assertIn("构造模型实例的工厂函数", digest)

    def test_upper_constants_present(self):
        digest = CodeValidator.interface_digest(SAMPLE_PY)
        self.assertIn("MAX_DEPTH = 4", digest)

    def test_no_bodies_leak(self):
        digest = CodeValidator.interface_digest(SAMPLE_PY)
        self.assertNotIn("do-not-leak", digest)
        self.assertNotIn("super().__init__", digest)
        self.assertNotIn("self.layers = ", digest)

    def test_private_module_constant_skipped(self):
        digest = CodeValidator.interface_digest(SAMPLE_PY)
        self.assertNotIn("_threshold", digest)

    def test_invalid_code_returns_empty(self):
        self.assertEqual(CodeValidator.interface_digest("def broken(:\n"), "")

    def test_digest_shorter_than_source(self):
        digest = CodeValidator.interface_digest(SAMPLE_PY)
        self.assertLess(len(digest), len(SAMPLE_PY))


class TestConfigDigest(unittest.TestCase):
    def test_paths_and_scalar_values(self):
        digest = CodeValidator.config_digest(SAMPLE_YAML)
        self.assertIn("project.name: demo", digest)
        self.assertIn("training.epochs: 10", digest)
        self.assertIn("data.versions.color.status: resolved", digest)

    def test_short_list_inlined(self):
        digest = CodeValidator.config_digest("a:\n  nums: [1, 2, 3]\n")
        self.assertIn("a.nums: [1, 2, 3]", digest)

    def test_long_list_summarized(self):
        digest = CodeValidator.config_digest(SAMPLE_YAML)
        self.assertIn("training.layers: [6 项]", digest)

    def test_invalid_yaml_returns_empty(self):
        self.assertEqual(CodeValidator.config_digest("a: [unclosed"), "")


class TestRelatedCodeDigest(unittest.TestCase):
    def setUp(self):
        brief = {
            "code_generation_plan": {
                "per_file_specs": [
                    {"file": "main.py", "depends_on": ["models/alexnet.py",
                                                      "config/config.yaml",
                                                      "README.md"]},
                ],
                "generation_order": [],
            },
        }
        self.builder = CodeContextBuilder(brief, step5_dir=None)
        self.generated = {
            "models/alexnet.py": SAMPLE_PY,
            "config/config.yaml": SAMPLE_YAML,
            "README.md": "# Title\n\nfull readme body\n",
        }

    def test_related_code_is_digest_not_source(self):
        related = self.builder._related_code(
            "main.py", {"depends_on": ["models/alexnet.py"]}, self.generated)
        digest = related["models/alexnet.py"]
        self.assertIn("class Alexnet", digest)
        self.assertNotIn("do-not-leak", digest)
        self.assertLess(len(digest), len(SAMPLE_PY))

    def test_yaml_dep_becomes_key_tree(self):
        related = self.builder._related_code(
            "main.py", {"depends_on": ["config/config.yaml"]}, self.generated)
        self.assertIn("training.epochs: 10", related["config/config.yaml"])

    def test_doc_deps_skipped(self):
        related = self.builder._related_code(
            "main.py", {"depends_on": ["README.md"]}, self.generated)
        self.assertNotIn("README.md", related)

    def test_missing_dep_skipped(self):
        related = self.builder._related_code(
            "main.py", {"depends_on": ["models/absent.py"]}, self.generated)
        self.assertEqual(related, {})


class TestTaskRouting(unittest.TestCase):
    def test_core_files_are_heavy(self):
        for f in ("models/alexnet.py", "models/nested/block.py", "training/train.py",
                  "data/dataset.py", "main.py", "run.py", "config/config.yaml"):
            self.assertEqual(task_type_for_file(f), "heavy", f)

    def test_aux_files_are_light(self):
        for f in ("utils/metrics.py", "evaluation/evaluate.py",
                  "models/__init__.py", "requirements.txt", "README.md"):
            self.assertEqual(task_type_for_file(f), "light", f)

    def test_generator_and_repairer_use_routing(self):
        # 生成/修复不再写死 heavy，而是按文件路由（验证调用点已接线）
        import inspect
        gen_src = inspect.getsource(CodeGenerator.generate)
        rep_src = inspect.getsource(CodeRepairer.repair)
        self.assertIn("task_type_for_file", gen_src)
        self.assertIn("task_type_for_file", rep_src)
        self.assertNotIn('task_type="heavy"', gen_src)
        self.assertNotIn('task_type="heavy"', rep_src)


class TestPromptWording(unittest.TestCase):
    def test_prompt_uses_digest_header(self):
        ctx = {
            "file": "main.py",
            "spec": {"purpose": "入口", "depends_on": ["models/alexnet.py"]},
            "project": {}, "tech_stack": {}, "training_config": {},
            "related_code": {"models/alexnet.py": "class Alexnet: ..."},
        }
        prompt = CodeGenerator()._build_prompt(ctx)
        self.assertIn("【依赖文件接口摘要（引用时保持一致）】", prompt)
        self.assertNotIn("【已生成的相关代码", prompt)

    def test_repair_prompt_uses_digest_header(self):
        ctx = {"related_code": {"models/alexnet.py": "class Alexnet: ..."}}
        prompt = CodeRepairer()._build_prompt(
            "main.py", "x = 1", ["err"], {"purpose": "入口"}, ctx)
        self.assertIn("【依赖文件接口摘要（引用必须与之一致）】", prompt)
        self.assertNotIn("【已生成的相关代码", prompt)


if __name__ == "__main__":
    unittest.main()
