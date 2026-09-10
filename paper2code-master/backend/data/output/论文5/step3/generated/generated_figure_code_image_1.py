import sys
import json
import numpy as np
import cv2

OCR_TEXT = """Forecast Time (t,
Prediction Intervals
Point Forecasts
Past Targets
Static
Covariates
Prediction
Model
Observed Inputs
Known Inputs"""


def load_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def get_visual_features(img):
    if img is None:
        return {"non_white_ratio": 0.0, "color_ratio": 0.0}

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return {"non_white_ratio": 0.0, "color_ratio": 0.0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_white_mask = gray < 240
    non_white_ratio = float(np.count_nonzero(non_white_mask)) / (h * w)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    color_ratio = float(np.count_nonzero(color_mask)) / (h * w)

    return {
        "non_white_ratio": non_white_ratio,
        "color_ratio": color_ratio,
    }


def normalize_text(text):
    return " ".join(text.lower().split())


def classify(ocr_text, visual):
    text = normalize_text(ocr_text)

    arch_strong = [
        "cnn",
        "transformer",
        "attention",
        "encoder",
        "decoder",
        "block",
        "embedding",
        "classifier",
        "feature extractor",
        "feature_extractor",
        "alexnet",
        "resnet",
        "vgg",
        "googlenet",
    ]

    arch_weak = [
        "prediction model",
        "observed inputs",
        "known inputs",
        "static covariates",
        "past targets",
        "point forecast",
        "prediction interval",
        "data flow",
        "pipeline",
        "model flow",
    ]

    exp_strong = [
        "accuracy",
        "loss",
        "epoch",
        "precision",
        "recall",
        "f1",
        "performance",
        "comparison",
        "ablation",
        "experimental result",
        "learning rate",
        "training",
        "validation",
        "test accuracy",
    ]

    exp_cn = [
        "准确率",
        "损失",
        "轮次",
        "精确率",
        "召回率",
    ]

    dataset_words = [
        "sample",
        "example",
        "dataset",
        "input example",
        "class image",
        "category",
    ]

    score_arch = 0
    score_exp = 0
    score_dataset = 0

    for keyword in arch_strong:
        if keyword in text:
            score_arch += 3

    for keyword in arch_weak:
        if keyword in text:
            score_arch += 1

    for keyword in exp_strong:
        if keyword in text:
            score_exp += 3

    for keyword in exp_cn:
        if keyword in text:
            score_exp += 2

    for keyword in dataset_words:
        if keyword in text:
            score_dataset += 2

    non_white_ratio = visual.get("non_white_ratio", 0.0)
    color_ratio = visual.get("color_ratio", 0.0)

    if 0.02 < non_white_ratio < 0.6 and color_ratio < 0.15:
        score_arch += 1

    if color_ratio > 0.25:
        score_exp += 1

    # 冲突优先级：experiment_result > architecture > dataset > other
    if score_exp >= 3:
        return "experiment_result"

    if score_arch >= 2:
        return "architecture"

    if score_dataset >= 2:
        return "dataset"

    return "other"


def extract_components(ocr_text):
    text = normalize_text(ocr_text)

    candidates = [
        ("prediction model", "Prediction Model"),
        ("feature extractor", "Feature Extractor"),
        ("classifier", "Classifier"),
        ("transformer encoder", "Transformer Encoder"),
        ("attention module", "Attention Module"),
        ("cnn", "CNN"),
        ("resnet", "ResNet"),
        ("alexnet", "AlexNet"),
        ("vgg", "VGG"),
        ("googlenet", "GoogLeNet"),
        ("decoder", "Decoder"),
        ("encoder", "Encoder"),
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("f1-score", "F1-score"),
        ("f1 score", "F1-score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]

    components = []
    for key, display in candidates:
        if key in text and display not in components:
            components.append(display)

    if "f1" in text and "F1-score" not in components:
        components.append("F1-score")

    return components


def main():
    try:
        image_path = sys.argv[1]
        img = load_image(image_path)
        visual = get_visual_features(img)

        image_type = classify(OCR_TEXT, visual)
        components = extract_components(OCR_TEXT)

        if image_type == "experiment_result":
            description = "展示实验对比结果或性能变化"
            code_relevant_information = "需要实现相关模型训练与评估流程，并输出对应指标"
        elif image_type == "architecture":
            description = "展示模型网络结构和数据流关系"
            code_relevant_information = "需要实现图中所示模型结构，包含输入处理、预测模型和输出生成"
        elif image_type == "dataset":
            description = "展示数据样本或输入示例"
            code_relevant_information = "需要准备或生成图中所示数据格式与样本"
        else:
            description = "无法确定图片类型"
            code_relevant_information = ""
            components = []

        result = {
            "type": image_type,
            "description": description,
            "important_components": components,
            "code_relevant_information": code_relevant_information,
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        fallback = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": "",
        }
        print(json.dumps(fallback, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()