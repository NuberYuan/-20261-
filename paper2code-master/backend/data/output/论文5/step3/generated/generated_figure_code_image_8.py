import os
import sys
import json
import numpy as np
import cv2


def load_image(image_path):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def read_ocr_text(image_path):
    if len(sys.argv) >= 3:
        return sys.argv[2]

    candidate = os.path.splitext(image_path)[0] + ".txt"
    if os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def simple_visual_features(img):
    if img is None or img.size == 0:
        return {}

    try:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        saturation = hsv[:, :, 1].astype(np.float32)
        mean_saturation = float(np.mean(saturation))
        mean_brightness = float(np.mean(gray))
        colorful_ratio = float(np.mean(saturation > 25))
        aspect = float(w) / max(float(h), 1.0)

        top_mean = float(np.mean(gray[: h // 2, :]))
        bottom_mean = float(np.mean(gray[h // 2:, :]))

        return {
            "aspect": aspect,
            "mean_saturation": mean_saturation,
            "mean_brightness": mean_brightness,
            "colorful_ratio": colorful_ratio,
            "top_bottom_diff": abs(top_mean - bottom_mean),
        }
    except Exception:
        return {}


def extract_components(ocr_text, category):
    text = ocr_text.lower()

    architecture_patterns = [
        "alexnet",
        "resnet",
        "vgg",
        "googlenet",
        "transformer encoder",
        "transformer decoder",
        "multi-head attention",
        "attention module",
        "attention",
        "encoder",
        "decoder",
        "embedding",
        "classifier",
        "feature extractor",
        "cnn",
        "rnn",
        "block",
        "module",
    ]

    experiment_patterns = [
        "alexnet",
        "resnet",
        "vgg",
        "googlenet",
        "transformer",
        "cnn",
        "rnn",
        "accuracy",
        "loss",
        "precision",
        "recall",
        "f1-score",
        "f1",
        "mean",
        "position index",
        "time step",
        "epoch",
        "auc",
        "attention",
    ]

    dataset_patterns = [
        "cifar-10",
        "cifar",
        "imagenet",
        "mnist",
        "coco",
        "voc",
        "dataset",
        "sample",
        "example",
        "input image",
        "class image",
        "label",
        "class",
    ]

    if category == "architecture":
        patterns = architecture_patterns
    elif category == "experiment_result":
        patterns = experiment_patterns
    elif category == "dataset":
        patterns = dataset_patterns
    else:
        patterns = []

    components = []

    for pattern in patterns:
        if pattern not in text:
            continue

        if pattern == "alexnet":
            component = "AlexNet"
        elif pattern == "resnet":
            component = "ResNet"
        elif pattern == "vgg":
            component = "VGG"
        elif pattern == "googlenet":
            component = "GoogLeNet"
        elif pattern == "transformer encoder":
            component = "Transformer Encoder"
        elif pattern == "transformer decoder":
            component = "Transformer Decoder"
        elif pattern == "multi-head attention":
            component = "Multi-Head Attention"
        elif pattern == "attention module":
            component = "Attention Module"
        elif pattern == "attention":
            component = "Attention"
        elif pattern == "encoder":
            component = "Encoder"
        elif pattern == "decoder":
            component = "Decoder"
        elif pattern == "embedding":
            component = "Embedding"
        elif pattern == "classifier":
            component = "Classifier"
        elif pattern == "feature extractor":
            component = "Feature Extractor"
        elif pattern == "cnn":
            component = "CNN"
        elif pattern == "rnn":
            component = "RNN"
        elif pattern == "accuracy":
            component = "Accuracy"
        elif pattern == "loss":
            component = "Loss"
        elif pattern == "precision":
            component = "Precision"
        elif pattern == "recall":
            component = "Recall"
        elif pattern == "f1-score":
            component = "F1-score"
        elif pattern == "f1":
            component = "F1-score"
        elif pattern == "mean":
            component = "Mean"
        elif pattern == "position index":
            component = "Position Index"
        elif pattern == "time step":
            component = "Time Step"
        elif pattern == "epoch":
            component = "Epoch"
        elif pattern == "auc":
            component = "AUC"
        elif pattern == "cifar-10":
            component = "CIFAR-10"
        elif pattern == "cifar":
            component = "CIFAR"
        elif pattern == "imagenet":
            component = "ImageNet"
        elif pattern == "mnist":
            component = "MNIST"
        elif pattern == "coco":
            component = "COCO"
        elif pattern == "voc":
            component = "VOC"
        elif pattern == "dataset":
            component = "Dataset"
        elif pattern == "sample":
            component = "Sample"
        elif pattern == "example":
            component = "Example"
        elif pattern == "input image":
            component = "Input Image"
        elif pattern == "class image":
            component = "Class Image"
        elif pattern == "label":
            component = "Label"
        elif pattern == "class":
            component = "Class"
        elif pattern == "block":
            component = "Block"
        elif pattern == "module":
            component = "Module"
        elif pattern == "transformer":
            component = "Transformer"
        else:
            component = pattern.title()

        if component not in components:
            components.append(component)

    return components


def classify(ocr_text, img):
    text = ocr_text.lower()
    text_compact = text.replace("\n", " ")

    experiment_keywords = [
        "accuracy",
        "loss",
        "epoch",
        "precision",
        "recall",
        "f1",
        "mean",
        "position index",
        "time step",
        "performance",
        "curve",
        "comparison",
        "baseline",
        "%",
        "percent",
        "value",
        "score",
        "equal",
        "approx",
        "t=1",
        "t=3",
        "t=5",
    ]

    architecture_keywords = [
        "cnn",
        "transformer",
        "attention",
        "encoder",
        "decoder",
        "block",
        "embedding",
        "classifier",
        "feature extractor",
        "alexnet",
        "resnet",
        "vgg",
        "googlenet",
    ]

    dataset_keywords = [
        "sample",
        "example",
        "dataset",
        "input image",
        "class image",
        "cifar",
        "mnist",
        "imagenet",
        "coco",
        "label",
    ]

    has_experiment = any(keyword in text_compact for keyword in experiment_keywords)
    has_architecture = any(keyword in text_compact for keyword in architecture_keywords)
    has_dataset = any(keyword in text_compact for keyword in dataset_keywords)

    visual = simple_visual_features(img)

    if has_experiment:
        category = "experiment_result"
        description = "展示实验曲线或性能对比，包含数值指标与位置或时间步变化"
        code_relevant = "需要实现实验评估、指标计算和曲线对比代码"
    elif has_architecture:
        category = "architecture"
        description = "展示模型网络结构和数据流关系"
        code_relevant = "需要实现图中对应网络模块及前向传播流程"
    elif has_dataset:
        category = "dataset"
        description = "展示数据样本或类别输入示例"
        code_relevant = "需要实现数据加载、预处理和样本组织代码"
    else:
        if not ocr_text.strip():
            category = "other"
            description = "无法确定图片类型"
            code_relevant = ""
        else:
            aspect = visual.get("aspect", 1.0)
            colorful_ratio = visual.get("colorful_ratio", 0.0)

            if aspect >= 2.0 and colorful_ratio < 0.2:
                category = "architecture"
                description = "根据版式和视觉信息推测为模型结构示意图"
                code_relevant = "需要根据论文上下文实现相应网络结构"
            else:
                category = "other"
                description = "无法确定图片类型"
                code_relevant = ""

    return {
        "type": category,
        "description": description,
        "important_components": extract_components(text_compact, category),
        "code_relevant_information": code_relevant,
    }


def main():
    result = {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": "",
    }

    try:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        if len(sys.argv) < 2:
            print(json.dumps(result, ensure_ascii=False))
            return

        image_path = sys.argv[1]
        ocr_text = read_ocr_text(image_path)
        img = load_image(image_path)

        if img is None:
            print(json.dumps(result, ensure_ascii=False))
            return

        result = classify(ocr_text, img)
    except Exception:
        result = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": "",
        }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()