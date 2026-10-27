"""Step7 平凡文件模板（确定性产出，不调 LLM）的单元测试。"""

import unittest

from step7.step7_main import Step7Runner
from step7.code_validator import CodeValidator


class TestTrivialTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = Step7Runner("template_test")

    def test_dispatch_only_covers_trivial_files(self):
        spec_map = {}
        self.assertIsNone(self.runner._template_trivial_file(
            "models/alexnet.py", {}, {}, [], spec_map))
        self.assertIsNotNone(self.runner._template_trivial_file(
            "requirements.txt", {}, {}, [], spec_map))
        self.assertIsNotNone(self.runner._template_trivial_file(
            "README.md", {}, {}, [], spec_map))
        self.assertIsNotNone(self.runner._template_trivial_file(
            "models/__init__.py", {}, {}, [], spec_map))

    def test_requirements_picks_up_spec_notes_packages(self):
        spec = {"notes": "包含 numpy, pillow, pyyaml, torch, torchvision, matplotlib。"}
        content = self.runner._template_requirements(spec)
        for pkg in ("numpy", "pillow", "pyyaml", "torch", "torchvision", "matplotlib"):
            self.assertIn(pkg, content.splitlines())
        # notes 未点名的包不应出现
        self.assertNotIn("pandas", content.splitlines())

    def test_requirements_baseline_packages_always_present(self):
        content = self.runner._template_requirements({})
        self.assertIn("numpy", content.splitlines())
        self.assertIn("pyyaml", content.splitlines())

    def test_readme_tree_matches_files_exactly(self):
        files = ["main.py", "config/config.yaml", "data/dataset.py",
                 "models/__init__.py", "models/alexnet.py", "requirements.txt", "README.md"]
        brief = {"project": {"paper_title": "T", "task": "image_classification"}}
        content = self.runner._template_readme(brief, files)
        # 每个计划文件都出现在树里
        for f in files:
            self.assertIn(f.rsplit("/", 1)[-1], content)
        # 不臆造常见目录/文件
        for bogus in ("setup.py", "configs/", "tests/", "experiments/", "src/"):
            self.assertNotIn(bogus, content)

    def test_readme_conclusions_not_hardcoded(self):
        content = self.runner._template_readme({"project": {}}, ["main.py"])
        for phrase in CodeValidator.BANNED_CONCLUSIONS:
            self.assertNotIn(phrase, content.lower())

    def test_package_init_exports_from_sibling_specs(self):
        spec_map = {
            "models/__init__.py": {"classes": []},
            "models/alexnet.py": {"classes": ["Alexnet"]},
            "models/googlenet.py": {"classes": ["Googlenet"]},
        }
        content = self.runner._template_package_init("models/__init__.py", spec_map)
        self.assertIn("from .alexnet import Alexnet", content)
        self.assertIn("from .googlenet import Googlenet", content)
        self.assertIn('__all__ = ["Alexnet", "Googlenet"]', content)

    def test_refresh_package_inits_aligns_with_generated_code(self):
        spec_map = {"models/__init__.py": {"classes": []}}
        generated = {
            "models/__init__.py": self.runner._template_package_init(
                "models/__init__.py", spec_map),
            "models/alexnet.py": "class Alexnet:\n    pass\n",
            "models/vgg.py": "class Vgg16:\n    pass\n",
        }
        validator = CodeValidator(all_files=list(generated))
        self.runner._refresh_package_inits(generated, validator)
        content = generated["models/__init__.py"]
        self.assertIn("from .alexnet import Alexnet", content)
        # spec 未声明、但代码里实际定义的类也被导出
        self.assertIn("from .vgg import Vgg16", content)

    def test_generated_templates_pass_validation(self):
        files = ["main.py", "models/__init__.py", "models/alexnet.py",
                 "requirements.txt", "README.md"]
        spec_map = {
            "models/__init__.py": {"classes": []},
            "models/alexnet.py": {"classes": ["Alexnet"]},
        }
        brief = {"project": {"paper_title": "T", "task": "image_classification"}}
        validator = CodeValidator(all_files=files)
        for f in files:
            content = self.runner._template_trivial_file(
                f, spec_map.get(f, {}), brief, files, spec_map)
            if content is None:
                continue
            val = validator.validate(f, content, spec_map.get(f, {}))
            self.assertEqual(val["status"], "passed", f"{f}: {val['errors']}")


if __name__ == "__main__":
    unittest.main()
