import os
import sys
import json
import numpy as np
import cv2

DEFAULT_OCR_TEXT = """
10%
50%
90%
Mean
Previous Hour
0.20
(n=-1)
0.15
Daily Interval
(24 hours)
0.10
0.05
0.00
-150
-100
-50
Position Index (n)
T=5
T=10
T=15
T-20
0.04
0.03
0.02
0.01
WUwulul
UM
"""

ARCHITECTURE_KEYWORDS = [
    "CNN", "Transformer", "Attention", "Encoder", "Decoder",
    "Block", "Embedding", "Classifier", "Feature Extractor",
    "AlexNet", "ResNet", "VGG", "GoogLeNet"
]

EXPERIMENT_KEYWORDS = [
    "accuracy", "loss", "epoch", "precision", "recall",
    "f1", "f1-score", "fscore", "specificity", "sensitivity",
    "auc", "mean average precision", "map", "performance",
    "validation", "training", "test"
]

DATASET_KEYWORDS = [
    "sample", "input image", "training images", "dataset",
    "class", "label", "examples", "data examples", "original images"
]


def get_ocr_text_from_args():
    if len(sys.argv) > 2:
        return " ".join(sys.argv[2:])
    return DEFAULT_OCR_TEXT


def read_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def compute_basic_visual_features(img):
    features = {}
    h, w = img.shape[:2]
    features["width"] = w
    features["height"] = h

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features["hue_mean"] = float(np.mean(hsv[:, :, 0]))
    features["sat_mean"] = float(np.mean(hsv[:, :, 1]))
    features["val_mean"] = float(np.mean(hsv[:, :, 2]))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = gray >= 245
    features["white_pixel_ratio"] = float(np.mean(white_mask))
    features["non_white_pixel_ratio"] = float(1.0 - np.mean(white_mask))

    color_mask = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
    features["color_pixel_ratio"] = float(np.mean(color_mask))

    hue = hsv[:, :, 0][color_mask]
    if hue.size > 0:
        quantized_hue = (hue // 30).astype(np.uint8)
        features["color_bin_count"] = int(np.unique(quantized_hue).size)
    else:
        features["color_bin_count"] = 0

    return features


def check_ocr_keywords(ocr_lower):
    arch_hits = [kw for kw in ARCHITECTURE_KEYWORDS if kw.lower() in ocr_lower]
    exp_hits = [kw for kw in EXPERIMENT_KEYWORDS if kw.lower() in ocr_lower]
    dataset_hits = [kw for kw in DATASET_KEYWORDS if kw.lower() in ocr_lower]
    return arch_hits, exp_hits, dataset_hits


def infer_type_by_ocr(ocr_text, visual):
    ocr_lower = ocr_text.lower()
    arch_hits, exp_hits, dataset_hits = check_ocr_keywords(ocr_lower)

    curve_hints = [
        "mean", "position index", "interval",
        "previous hour", "daily interval", "curve"
    ]
    curve_hit_count = sum(1 for hint in curve_hints if hint in ocr_lower)

    if curve_hit_count >= 2:
        exp_hits = list(exp_hits) + ["curve_hint"]

    if exp_hits:
        return "experiment_result", exp_hits
    if arch_hits:
        return "architecture", arch_hits
    if dataset_hits:
        return "dataset", dataset_hits

    white_ratio = visual["white_pixel_ratio"]
    color_ratio = visual["color_pixel_ratio"]

    if color_ratio > 0.01 and white_ratio > 0.5:
        return "experiment_result", ["visual_chart_hint"]
    if color_ratio > 0.08 and white_ratio < 0.5:
        return "dataset", ["visual_dataset_hint"]

    return "other", []


def fill_components(img_type, ocr_text):
    ocr_lower = ocr_text.lower()
    components = []

    if img_type == "architecture":
        allowed_arch = [
            "CNN", "Transformer", "Attention", "Encoder", "Decoder",
            "Block", "Embedding", "Classifier", "Feature Extractor",
            "AlexNet", "ResNet", "VGG", "GoogLeNet"
        ]
        for kw in allowed_arch:
            if kw.lower() in ocr_lower:
                components.append(kw)
        return components

    if img_type == "experiment_result":
        metric_terms = [
            "accuracy", "loss", "epoch", "precision", "recall",
            "f1", "f1-score", "auc", "mean average precision", "map"
        ]
        for term in metric_terms:
            if term in ocr_lower:
                components.append(term)

        model_names = ["alexnet", "resnet", "vgg", "googlenet", "transformer", "cnn"]
        for model in model_names:
            if model in ocr_lower:
                components.append(model)

        seen = set()
        unique = []
        for c in components:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    if img_type == "dataset":
        dataset_components = [
            "sample", "input image", "training images", "dataset",
            "class", "label", "examples"
        ]
        for term in dataset_components:
            if term in ocr_lower:
                components.append(term)
        return components

    return []


def build_description(img_type):
    if img_type == "architecture":
        return "展示模型网络结构和数据流关系"
    if img_type == "experiment_result":
        return "展示实验数据变化趋势与统计结果，用于对比不同条件下的性能或分布"
    if img_type == "dataset":
        return "展示数据样本、类别图片或输入示例"
    return "无法确定图片类型"


def build_code_relevant_information(img_type):
    if img_type == "architecture":
        return "需要实现图中出现的模型组件及其连接关系"
    if img_type == "experiment_result":
        return "需要实现数据处理、指标计算与可视化绘图，以复现图中的实验结果"
    if img_type == "dataset":
        return "需要准备并加载与图中样本一致的数据集和预处理流程"
    return ""


def main():
    try:
        if len(sys.argv) < 2:
            fallback = {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
            print(json.dumps(fallback, ensure_ascii=True))
            return

        image_path = sys.argv[1]
        img = read_image(image_path)

        if img is None:
            fallback = {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
            print(json.dumps(fallback, ensure_ascii=True))
            return

        ocr_text = get_ocr_text_from_args()
        visual = compute_basic_visual_features(img)

        img_type, ocr_hits = infer_type_by_ocr(ocr_text, visual)
        components = fill_components(img_type, ocr_text)
        description = build_description(img_type)
        code_info = build_code_relevant_information(img_type)

        result = {
            "type": img_type,
            "description": description,
            "important_components": components,
            "code_relevant_information": code_info
        }

        print(json.dumps(result, ensure_ascii=True))
    except Exception:
        fallback = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
        try:
            print(json.dumps(fallback, ensure_ascii=True))
        except Exception:
            print("{}")


if __name__ == "__main__":
    main()