"""
Step8.2: 资源检查器 (Resource Checker)

用程序实际检测当前机器的真实环境（不是 LLM 猜测）。Step8.md 第六节明确要求：

    AI / 解析器   → 只负责理解实验"需要什么"
    程序          → 只负责检查当前机器"真实有什么"

检测项（8 个资源维度，通用化，不偏向图片/CNN/GPU）：
    - CPU         核数：os.cpu_count()
    - RAM         总量/可用：Windows ctypes / Linux /proc/meminfo（不依赖 psutil）
    - GPU         是否有：torch.cuda.is_available()（GPU 非必需，仅在实验要求时校验）
    - GPU Memory  显存：torch.cuda.get_device_properties(...)
    - Disk        可用空间：shutil.disk_usage(...)
    - Dataset     目录是否存在 + 数据类型/样本数/总容量（过大 → 警告，非 SKIP）
    - Python      版本：sys.version_info
    - Dependencies 依赖包可导入性：importlib.util.find_spec(...)
"""

import ctypes
import importlib.util
import os
import shutil
import sys

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch 未安装时降级为纯环境信息
    torch = None
    _HAS_TORCH = False

# 数据集"过大"的判定：以总容量（GB）而非样本数量为准 —— 60000 个 28×28 图片
# 可能比 5000 个 4096×4096 图片轻得多。过大只产生 warning（large_dataset），不 SKIP。
LARGE_DATASET_SIZE_GB = 5.0

# 数据集扫描上限（文件数）：避免超大目录遍历过久。
_SCAN_FILE_LIMIT = 20000

# 常见 CNN 架构参数量参考值（约值，单位：参数个数），用于粗略显存估算。
# best-effort 参考：数值为常见实现（如 torchvision 默认配置）的近似参数量，非精确值。
# 模型不在表中时 estimate_gpu_memory_gb 返回 None，check_gpu 按「无法估算则视为满足」处理；
# RNN/VLM/Transformer 等参数量随超参变化大，不在此表，自然回退为不估算。
_MODEL_PARAMS = {
    "alexnet": 61_000_000,
    "vgg11": 133_000_000,
    "vgg13": 133_000_000,
    "vgg16": 138_000_000,
    "vgg19": 144_000_000,
    "googlenet": 6_800_000,
    "inception_v3": 27_000_000,
    "resnet18": 11_700_000,
    "resnet34": 21_800_000,
    "resnet50": 25_600_000,
    "resnet101": 44_500_000,
    "resnet152": 60_000_000,
    "densenet121": 8_000_000,
    "densenet169": 14_000_000,
    "densenet201": 20_000_000,
    "densenet161": 28_700_000,
    "mobilenet_v2": 3_500_000,
    "mobilenet_v3_large": 5_500_000,
    "mobilenet_v3_small": 2_500_000,
    "efficientnet_b0": 5_300_000,
    "efficientnet_b1": 7_800_000,
    "squeezenet1_0": 1_250_000,
    "squeezenet1_1": 1_250_000,
}

# 归一化后的参数表：key 去除非字母数字字符并转小写，用于对上游 model_design 产出的
# 大小写/命名变体（如 "Alexnet" / "GoogLeNet" / "MobileNetV2"）做容错匹配。
_MODEL_PARAMS_NORMALIZED = {
    "".join(ch for ch in k if ch.isalnum()): v
    for k, v in _MODEL_PARAMS.items()
}

# 数据类型识别：按扩展名归类为 image/csv/text/json/graph/other。
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".ppm", ".webp")
_CSV_EXTS = (".csv",)
_TEXT_EXTS = (".txt", ".tsv", ".md", ".text", ".log")
_JSON_EXTS = (".json", ".jsonl", ".ndjson")
# .npz/.pt 是 PyG/DGL 等图神经网络库常见的图数据存储格式
_GRAPH_EXTS = (".graphml", ".gml", ".gexf", ".edgelist", ".edges", ".edge", ".mtx", ".npz", ".pt")

# 非外部数据集来源：这些情况下无需检查数据集目录（数据集检查不适用）。
_NON_EXTERNAL_DATA_SOURCES = {"synthetic", "environment", "none"}

# 依赖探针：检测这些常见包是否可导入（仅探测环境事实，不是需求判断）。
_DEPENDENCY_PROBES = (
    "torch", "numpy", "pandas", "sklearn", "scipy", "matplotlib",
    "PIL", "yaml", "tqdm", "cv2", "huggingface_hub", "transformers",
)

# 发行名（requirements.txt / config system.dependencies 常见写法）→ 可导入模块名。
# importlib.util.find_spec 需要导入名，发行名（scikit-learn/Pillow/PyYAML 等）会误判缺失。
_REQ_TO_IMPORT = {
    "scikit-learn": "sklearn",
    "scikit_learn": "sklearn",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "opencv": "cv2",
    "huggingface-hub": "huggingface_hub",
    "python-dotenv": "dotenv",
}


class ResourceChecker:
    """检测真实运行环境，供 Scheduler 做 RUN/SKIP 决策。"""

    def __init__(self):
        self.environment = self._detect_environment()

    def _detect_environment(self) -> dict:
        gpu_available = False
        gpu_memory_gb = None
        torch_version = None
        if _HAS_TORCH:
            torch_version = str(torch.__version__)
            try:
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    props = torch.cuda.get_device_properties(0)
                    gpu_memory_gb = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                gpu_available = False
        ram_total_gb, ram_available_gb = self._detect_memory()
        return {
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "cpu_count": os.cpu_count(),
            "ram_total_gb": ram_total_gb,
            "ram_available_gb": ram_available_gb,
            "disk_free_gb": self._disk_free_gb(),
            "torch_available": _HAS_TORCH,
            "torch_version": torch_version,
            "gpu_available": gpu_available,
            "gpu_memory_gb": gpu_memory_gb,
            "dependencies": self._probe_dependencies(),
        }

    @staticmethod
    def _disk_free_gb():
        """当前工作目录所在磁盘的可用空间（GB），失败返回 None。"""
        try:
            return round(shutil.disk_usage(".").free / (1024 ** 3), 2)
        except Exception:
            return None

    @staticmethod
    def _detect_memory():
        """返回 (total_gb, available_gb)，不依赖 psutil。

        Windows 用 ctypes GlobalMemoryStatusEx，Linux 读 /proc/meminfo；
        其他平台或失败时返回 (None, None)。
        """
        total = available = None
        try:
            if os.name == "nt":
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                ms = MEMORYSTATUSEX()
                ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
                total = round(ms.ullTotalPhys / (1024 ** 3), 2)
                available = round(ms.ullAvailPhys / (1024 ** 3), 2)
            elif sys.platform.startswith("linux"):
                info = {}
                with open("/proc/meminfo", encoding="utf-8") as f:
                    for line in f:
                        k, _, v = line.partition(":")
                        info[k.strip()] = int(v.strip().split()[0])
                total = round(info.get("MemTotal", 0) / (1024 ** 2), 2)
                available = round(
                    info.get("MemAvailable", info.get("MemFree", 0)) / (1024 ** 2), 2
                )
        except Exception:
            pass
        return total, available

    @staticmethod
    def _probe_dependencies() -> dict:
        """检测常见依赖包是否可导入（探测环境事实，非需求判断）。"""
        return {
            pkg: importlib.util.find_spec(pkg) is not None
            for pkg in _DEPENDENCY_PROBES
        }

    # ---- 数据集检查 ----

    def check_dataset(self, exp: dict) -> dict:
        """返回数据集可用性事实 + 类型/规模信息（sufficient 供调度器统一判断）。

        硬性不满足（SKIP）：仅 dataset_unresolved / dataset_unavailable。
        "过大"不再 SKIP，改为 warning（large_dataset）——样本数量不能代表实际容量。
        """
        warnings = []
        data_source = exp.get("data_source") or "external_dataset"
        if data_source in _NON_EXTERNAL_DATA_SOURCES:
            return {
                "sufficient": True,
                "reason": "not_required",
                "detail": f"数据集不适用（data_source={data_source}）",
                "type": None, "samples": None, "total_size_gb": None,
                "scan": None,
                "large": False, "warnings": warnings,
            }
        ds = exp.get("dataset", {}) or {}
        if ds.get("status") == "unresolved":
            return {
                "sufficient": False,
                "reason": "dataset_unresolved",
                "detail": ds.get("reason") or "数据来源在论文/设计中未定义",
                "type": None, "samples": None, "total_size_gb": None,
                "scan": None,
                "large": False, "warnings": warnings,
            }
        # auto 获取方式：数据集可自动下载/随库内置（MNIST/sklearn/HF），无需检查本地目录。
        acquisition = exp.get("acquisition") or "local_path"
        if acquisition == "auto":
            return {
                "sufficient": True,
                "reason": "auto_download",
                "detail": "数据集可自动下载/随库内置（acquisition=auto），本地不扫描",
                "type": None, "samples": None, "total_size_gb": None,
                "scan": None,
                "large": False, "warnings": warnings,
            }
        path = ds.get("path")
        if not path or not os.path.isdir(str(path)):
            return {
                "sufficient": False,
                "reason": "dataset_unavailable",
                "detail": f"目录不存在: {path}",
                "type": None, "samples": None, "total_size_gb": None,
                "scan": None,
                "large": False, "warnings": warnings,
            }
        scan = self._scan_dataset(str(path))
        dtype = scan["type"]
        samples = scan["samples"]
        total_size_gb = scan["total_size_gb"]
        large = total_size_gb > LARGE_DATASET_SIZE_GB
        if large:
            warnings.append("large_dataset")
        detail = f"{dtype} 数据集，{samples} 样本，{total_size_gb}GB"
        if scan["truncated"]:
            detail += f"（扫描截断，仅前 {scan['scanned_files']} 个文件）"
        return {
            "sufficient": True,
            "reason": None,
            "detail": detail,
            "type": dtype,
            "samples": samples,
            "total_size_gb": total_size_gb,
            "scan": {
                "scanned_files": scan["scanned_files"],
                "scan_limit": scan["scan_limit"],
                "truncated": scan["truncated"],
            },
            "large": large,
            "warnings": warnings,
        }

    @staticmethod
    def _classify_type(ext: str) -> str:
        """按扩展名归类数据类型：image/csv/text/json/graph/other。"""
        if ext in _IMAGE_EXTS:
            return "image"
        if ext in _CSV_EXTS:
            return "csv"
        if ext in _TEXT_EXTS:
            return "text"
        if ext in _JSON_EXTS:
            return "json"
        if ext in _GRAPH_EXTS:
            return "graph"
        return "other"

    def _scan_dataset(self, path: str) -> dict:
        """扫描数据集目录，返回 {type, samples, total_size_gb, scanned_files, scan_limit, truncated}。

        type          = 目录中占多数的数据类型（image/csv/text/json/graph/other）
        samples       = 该类型文件数（image 即图片数）；truncated=True 时仅为已扫描部分
        total_size_gb = 已扫描文件的累计容量（GB）；truncated=True 时为部分容量
        有界扫描（_SCAN_FILE_LIMIT），避免超大目录遍历过久；截断时明确标记 truncated。
        """
        type_counts = {}
        total_bytes = 0
        scanned = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                typ = self._classify_type(os.path.splitext(f)[1].lower())
                type_counts[typ] = type_counts.get(typ, 0) + 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
                scanned += 1
                if scanned >= _SCAN_FILE_LIMIT:
                    break
            if scanned >= _SCAN_FILE_LIMIT:
                break
        truncated = scanned >= _SCAN_FILE_LIMIT
        if not type_counts:
            return {
                "type": "other", "samples": 0, "total_size_gb": 0.0,
                "scanned_files": 0, "scan_limit": _SCAN_FILE_LIMIT, "truncated": False,
            }
        dominant = max(type_counts, key=type_counts.get)
        return {
            "type": dominant,
            "samples": type_counts[dominant],
            "total_size_gb": round(total_bytes / (1024 ** 3), 3),
            "scanned_files": scanned,
            "scan_limit": _SCAN_FILE_LIMIT,
            "truncated": truncated,
        }

    # ---- GPU 检查 ----

    def check_gpu(self, exp: dict) -> dict:
        """返回 GPU 是否满足该实验（程序检测，非猜测）。

        GPU 不可用（gpu_unavailable）不再硬性 SKIP：生成的训练代码普遍带
        CUDA→CPU 回退（train.py 检测不到 CUDA 时自动改用 CPU），降级运行
        只是变慢、不致失败，故记 warning 后放行（CPU-degraded RUN）。
        显存不足（insufficient_gpu_memory）基于粗略显存估算，同样只产生
        warning，不 SKIP —— 估算值并非精确显存需求，不能据此硬性跳过。
        """
        warnings = []
        gpu_required = bool(exp.get("gpu_required"))
        gpu_available = bool(self.environment.get("gpu_available"))
        if gpu_required and not gpu_available:
            warnings.append("gpu_unavailable_cpu_fallback")
            return {
                "sufficient": True,
                "reason": None,
                "detail": (
                    "实验要求 GPU 但当前无可用 GPU；降级为 CPU 运行"
                    "（慢、结果可能有微小数值差异），生成代码的 CUDA→CPU 回退生效"
                ),
                "rough_gpu_memory_gb": None,
                "warnings": warnings,
            }
        if not gpu_required:
            return {
                "sufficient": True,
                "reason": None,
                "detail": "不强制要求 GPU（可 CPU 运行）",
                "rough_gpu_memory_gb": None,
                "warnings": warnings,
            }
        # 有 GPU：粗略显存估算，不足仅警告（非精确需求）
        estimated = self.estimate_gpu_memory_gb(exp)
        available = self.environment.get("gpu_memory_gb")
        if estimated is not None and available is not None and estimated > available:
            warnings.append("insufficient_gpu_memory")
            return {
                "sufficient": True,
                "reason": None,
                "detail": (
                    f"粗略估算需 {estimated:.2f}GB，可用 {available:.2f}GB"
                    f"（估算非精确，仅警告，不跳过）"
                ),
                "rough_gpu_memory_gb": estimated,
                "available_gpu_memory_gb": available,
                "warnings": warnings,
            }
        detail = "GPU 显存满足要求"
        if estimated is not None:
            detail += f"（粗略估算 {estimated:.2f}GB）"
        return {
            "sufficient": True,
            "reason": None,
            "detail": detail,
            "rough_gpu_memory_gb": estimated,
            "warnings": warnings,
        }

    @staticmethod
    def estimate_gpu_memory_gb(exp: dict):
        """粗略显存估算（GB）：权重 + 梯度 + 动量 + 激活缓冲区 ≈ 参数量 * 4 字节 * 4。

        明确标记为 rough_estimation：未计入 batch_size / input_size / optimizer 等因素，
        结果仅作参考，不得当作精确显存需求（check_gpu 因此只降级为 warning，不 SKIP）。
        """
        model = exp.get("model")
        if not model:
            return None
        # 大小写/命名变体容错：先用归一化 key 查表，查不到则视为「无法估算」
        key = "".join(ch for ch in str(model) if ch.isalnum()).lower()
        params = _MODEL_PARAMS_NORMALIZED.get(key)
        if params is None:
            return None
        bytes_ = params * 4 * 4
        return round(bytes_ / (1024 ** 3), 2)

    # ---- CPU / RAM / Disk / Python / Dependencies ----
    # 这些检查只在实验声明了对应需求（exp 里的可选字段，来自 config 的 system 段）时
    # 才做「需求 vs 事实」比较；未声明则一律 sufficient=True（不做硬性门控）。

    def check_cpu(self, exp: dict) -> dict:
        required = exp.get("cpu_cores")
        if not required:
            return {"sufficient": True, "reason": None, "detail": "未声明 CPU 核数需求"}
        available = self.environment.get("cpu_count")
        if available is None:
            return {"sufficient": True, "reason": None, "detail": "无法检测 CPU 核数"}
        if available < required:
            return {"sufficient": False, "reason": "insufficient_cpu",
                    "detail": f"需 {required} 核，可用 {available} 核"}
        return {"sufficient": True, "reason": None,
                "detail": f"{available} 核满足需求（需 {required}）"}

    def check_ram(self, exp: dict) -> dict:
        required = exp.get("ram_gb")
        if not required:
            return {"sufficient": True, "reason": None, "detail": "未声明内存需求"}
        available = self.environment.get("ram_available_gb")
        if available is None:
            available = self.environment.get("ram_total_gb")
        if available is None:
            return {"sufficient": True, "reason": None, "detail": "无法检测内存"}
        if available < required:
            return {"sufficient": False, "reason": "insufficient_ram",
                    "detail": f"需 {required}GB，可用 {available}GB"}
        return {"sufficient": True, "reason": None,
                "detail": f"{available}GB 可用满足需求（需 {required}GB）"}

    def check_disk(self, exp: dict) -> dict:
        required = exp.get("disk_gb")
        if not required:
            return {"sufficient": True, "reason": None, "detail": "未声明磁盘空间需求"}
        available = self.environment.get("disk_free_gb")
        if available is None:
            return {"sufficient": True, "reason": None, "detail": "无法检测磁盘空间"}
        if available < required:
            return {"sufficient": False, "reason": "insufficient_disk",
                    "detail": f"需 {required}GB，可用 {available}GB"}
        return {"sufficient": True, "reason": None,
                "detail": f"{available}GB 可用满足需求（需 {required}GB）"}

    def check_python(self, exp: dict) -> dict:
        required = exp.get("python_version")  # 最低版本，如 "3.8"
        if not required:
            return {"sufficient": True, "reason": None, "detail": "未声明 Python 版本需求"}
        cur = self.environment.get("python_version")
        if cur is None:
            return {"sufficient": True, "reason": None, "detail": "无法检测 Python 版本"}
        if self._version_lt(cur, required):
            return {"sufficient": False, "reason": "python_version_too_low",
                    "detail": f"需 ≥{required}，当前 {cur}"}
        return {"sufficient": True, "reason": None,
                "detail": f"Python {cur} 满足需求（需 ≥{required}）"}

    @staticmethod
    def _import_name(req: str) -> str:
        """把发行名解析为可导入模块名（scikit-learn→sklearn、Pillow→PIL 等），
        其余小写原样返回。"""
        name = str(req).strip().lower()
        return _REQ_TO_IMPORT.get(name, name)

    def check_dependencies(self, exp: dict) -> dict:
        required = exp.get("dependencies") or []
        if not required:
            return {"sufficient": True, "reason": None, "detail": "未声明依赖包需求"}
        # 用导入名探测，但报告缺失时用原始声明名（用户按声明名 pip install 更直观）。
        missing = [p for p in required
                   if importlib.util.find_spec(self._import_name(p)) is None]
        if missing:
            return {"sufficient": False, "reason": "missing_dependencies",
                    "detail": f"缺少依赖: {', '.join(missing)}"
                              f"（请 pip install -r requirements.txt 后重跑）"}
        return {"sufficient": True, "reason": None,
                "detail": f"{len(required)} 个依赖均可用"}

    @staticmethod
    def _version_lt(a: str, b: str) -> bool:
        """比较语义化版本 a < b（容忍 "3.8" 缺少 micro）。"""
        def parts(v):
            try:
                return [int(x) for x in str(v).split(".")[:3]]
            except (ValueError, TypeError):
                return []
        pa, pb = parts(a), parts(b)
        while len(pa) < 3:
            pa.append(0)
        while len(pb) < 3:
            pb.append(0)
        return pa < pb

    def check_all(self, exp: dict) -> dict:
        """按优先级顺序汇总所有资源检查（ExperimentScheduler 据此决定 RUN/SKIP）。

        顺序即 skip_reasons 的优先级：dataset > gpu > cpu > ram > disk > python > dependencies。
        GPU 不可用不构成 SKIP：降级 CPU 运行，仅记 warning（gpu_unavailable_cpu_fallback）。
        """
        return {
            "dataset": self.check_dataset(exp),
            "gpu": self.check_gpu(exp),
            "cpu": self.check_cpu(exp),
            "ram": self.check_ram(exp),
            "disk": self.check_disk(exp),
            "python": self.check_python(exp),
            "dependencies": self.check_dependencies(exp),
        }
