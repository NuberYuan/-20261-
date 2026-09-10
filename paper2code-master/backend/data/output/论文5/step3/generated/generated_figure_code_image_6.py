import sys
import json
import numpy as np
import cv2

OCR_TEXT = """
10%
50%
90%
Mean
Previous Hour
0.3
Daily Interval
(n=-1)
(24 hours)
0.2
0.1
0.0
-150
-100
-50
Position Index (n)
Forecast
0=5
T=10
T=15
T=20
Time
0.15
0.10
0.05
0.00
HHHULLu
"""

ARCH_KEYWORDS = [
    "cnn", "transformer", "attention", "encoder", "decoder", "block",
    "embedding", "classifier", "feature extractor", "alexnet",
    "resnet", "vgg", "googlenet", "convolution", "pooling",
    "network", "pipeline", "module", "layer"
]

EXP_STRONG_KEYWORDS = [
    "accuracy", "loss", "epoch", "precision", "recall", "f1",
    "f1-score", "validation", "training", "test", "performance",
    "experiment", "curve", "comparison", "ablation", "baseline",
    "benchmark", "batch size", "learning rate"
]

EXP_CURVE_KEYWORDS = [
    "forecast", "mean", "previous hour", "daily interval",
    "position index", "time", "percentage", "value", "trend"
]

DATASET_KEYWORDS = [
    "dataset", "sample", "example", "input image", "class image",
    "category", "label", "cifar", "mnist", "imagenet", "ground truth"
]

MODEL_MAP = {
    "alexnet": "AlexNet",
    "resnet": "ResNet",
    "vgg": "VGG",
    "googlenet": "GoogLeNet",
    "transformer": "Transformer",
    "attention": "Attention",
    "encoder": "Encoder",
    "decoder": "Decoder",
    "embedding": "Embedding",
    "classifier": "Classifier",
    "feature extractor": "Feature Extractor",
    "cnn": "CNN",
    "lstm": "LSTM",
    "rnn": "RNN",
    "unet": "U-Net",
    "bert": "BERT"
}

METRIC_MAP = {
    "accuracy": "Accuracy",
    "loss": "Loss",
    "precision": "Precision",
    "recall": "Recall",
    "f1-score": "F1-score",
    "f1": "F1-score",
    "mean": "Mean",
    "forecast": "Forecast",
    "auc": "AUC",
    "map": "mAP",
    "iou": "IoU",
    "epoch": "Epoch"
}


def read_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def get_visual_features(img):
    if img is None:
        return {}
    if img.size == 0:
        return {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    white_ratio = float((val > 230).mean())
    colored_ratio = float((sat > 80).mean())

    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    quantized = small // 32 * 32
    unique_colors = np.unique(quantized.reshape(-1, quantized.shape[2]), axis=0)

    return {
        "white_ratio": white_ratio,
        "colored_ratio": colored_ratio,
        "color_count": int(len(unique_colors))
    }


def classify_image(ocr_text, visual):
    lower = ocr_text.lower()

    exp_strong_score = 0
    for keyword in EXP_STRONG_KEYWORDS:
        if keyword in lower:
            exp_strong_score += 3

    exp_curve_score = 0
    for keyword in EXP_CURVE_KEYWORDS:
        if keyword in lower:
            exp_curve_score += 1

    arch_score = 0
    for keyword in ARCH_KEYWORDS:
        if keyword in lower:
            arch_score += 2

    dataset_score = 0
    for keyword in DATASET_KEYWORDS:
        if keyword in lower:
            dataset_score += 2

    if any(ch in ocr_text for ch in ["%", "=", "0.", "-"]):
        exp_curve_score += 1

    if visual:
        white_ratio = visual.get("white_ratio", 0)
        colored_ratio = visual.get("colored_ratio", 0)
        color_count = visual.get("color_count", 0)

        if white_ratio > 0.45 and 0.001 < colored_ratio < 0.25:
            exp_curve_score += 1

        if color_count > 16:
            arch_score += 1

    if exp_strong_score > 0 or exp_curve_score >= 2:
        return "experiment_result"

    if arch_score > 0:
        return "architecture"

    if dataset_score > 0:
        return "dataset"

    return "other"


def extract_components(category, text):
    lower = text.lower()
    candidates = []

    if category == "architecture":
        keys = list(MODEL_MAP.keys())
    elif category == "experiment_result":
        keys = list(MODEL_MAP.keys()) + list(METRIC_MAP.keys())
    else:
        return []

    for key in keys:
        pos = lower.find(key)
        if pos != -1:
            name = MODEL_MAP.get(key, METRIC_MAP.get(key))
            if name:
                candidates.append((pos, name))

    seen = set()
    components = []
    for _, name in sorted(candidates, key=lambda item: item[0]):
        if name.lower() not in seen:
            seen.add(name.lower())
            components.append(name)

    return components


def build_description(category):
    mapping = {
        "architecture": "展示模型网络结构和数据流关系",
        "experiment_result": "展示实验曲线和性能变化趋势",
        "dataset": "展示数据样本、类别图片或输入示例",
        "other": "无法确定图片类型"
    }
    return mapping.get(category, "无法确定图片类型")


def build_code_info(category, components):
    if category == "architecture":
        return "需要实现特征提取模块、编码器或分类头等网络结构"

    if category == "experiment_result":
        if components:
            return "需要实现实验评估、指标计算与可视化，关注" + "、".join(components)
        return "需要实现实验评估、指标计算与结果可视化"

    if category == "dataset":
        return "需要准备数据加载、预处理与输入样本构建"

    return ""


def main():
    image_path = None
    visual = {}

    try:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = r"D:\applications\paper2code-master\paper2code-master\backend\data\output\论文5\step1\images\image_6.png"

        img = read_image(image_path)
        visual = get_visual_features(img)
    except Exception:
        visual = {}

    try:
        category = classify_image(OCR_TEXT, visual)
        description = build_description(category)
        components = extract_components(category, OCR_TEXT)
        code_info = build_code_info(category, components)

        result = {
            "type": category,
            "description": description,
            "important_components": components,
            "code_relevant_information": code_info
        }

        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        fallback = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
        print(json.dumps(fallback, ensure_ascii=False))


if __name__ == "__main__":
    main()