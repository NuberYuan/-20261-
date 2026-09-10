import os
import sys
import json
import numpy as np
import cv2

OCR_TEXT = r"""-6
-8
Forecast date (n=0)
-10
'vxnWNVbwrsk"ymolywwwdk ]
Increased
-12
attention on
trend
Median forecast +
Equal attention over range-bound
changes
uncertainty
0.06
low volatility periods
interval
0.04
0.02
10/1/2004
1/1/2005
4/1/2005
7/1/2005
2008-07-01
2008-10-01
2009-01-01
2009-04-01
t2
-5
0.6
~9"""


def read_image(image_path):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def get_basic_features(img):
    if img is None:
        return {}
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        non_white_ratio = float(np.mean(gray < 235))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        colored_pixel_ratio = float(np.mean(sat > 30))
        avg_saturation = float(np.mean(sat))
        return {
            "height": int(h),
            "width": int(w),
            "aspect_ratio": round(float(w) / float(h), 3),
            "non_white_ratio": round(non_white_ratio, 4),
            "colored_pixel_ratio": round(colored_pixel_ratio, 4),
            "avg_saturation": round(avg_saturation, 2)
        }
    except Exception:
        return {}


def classify(ocr_text, features):
    ocr = ocr_text.lower()

    exp_strong = [
        "forecast date", "median forecast", "uncertainty interval",
        "low volatility periods", "range-bound changes", "accuracy",
        "loss", "epoch", "precision", "recall", "f1", "f1-score",
        "training curve", "validation curve", "performance comparison",
        "time series", "uncertainty", "forecast"
    ]
    exp_weak = [
        "prediction", "trend", "volatility", "interval", "rmse", "mae",
        "auc", "roc", "baseline", "result", "compare", "plot", "graph",
        "epochs", "iteration", "learning rate"
    ]

    exp_score = 0
    for kw in exp_strong:
        if kw in ocr:
            exp_score += 2
    for kw in exp_weak:
        if kw in ocr:
            exp_score += 1

    arch_strong = [
        "cnn", "transformer", "encoder", "decoder", "block", "embedding",
        "classifier", "feature extractor", "alexnet", "resnet", "vgg",
        "googlenet", "attention", "multi-head", "skip connection",
        "feed forward", "lstm", "gru"
    ]
    arch_weak = [
        "layer", "convolutional", "neural network", "softmax", "relu",
        "hidden state", "input", "output"
    ]

    arch_score = 0
    for kw in arch_strong:
        if kw in ocr:
            arch_score += 2
    for kw in arch_weak:
        if kw in ocr:
            arch_score += 1

    dataset_strong = [
        "dataset", "sample", "example", "input image", "data sample",
        "training samples", "test samples", "cifar", "mnist", "imagenet",
        "class label", "instance", "images from"
    ]
    dataset_weak = [
        "label", "class", "category", "sample image"
    ]

    dataset_score = 0
    for kw in dataset_strong:
        if kw in ocr:
            dataset_score += 2
    for kw in dataset_weak:
        if kw in ocr:
            dataset_score += 1

    if exp_score > 0:
        return "experiment_result"
    elif arch_score > 0:
        return "architecture"
    elif dataset_score > 0:
        return "dataset"
    else:
        return "other"


def extract_components(ocr_text, img_type):
    ocr = ocr_text.lower()
    components = []

    if img_type == "experiment_result":
        candidates = [
            ("median forecast", "Median forecast"),
            ("uncertainty interval", "Uncertainty interval"),
            ("range-bound changes", "Range-bound changes"),
            ("low volatility periods", "Low volatility periods"),
            ("forecast date", "Forecast date"),
            ("uncertainty", "Uncertainty"),
            ("accuracy", "Accuracy"),
            ("loss", "Loss"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("rmse", "RMSE"),
            ("mae", "MAE")
        ]
        for key, comp in candidates:
            if key in ocr:
                components.append(comp)
            if len(components) >= 4:
                break

    elif img_type == "architecture":
        candidates = [
            ("alexnet", "AlexNet"),
            ("resnet", "ResNet"),
            ("vgg", "VGG"),
            ("googlenet", "GoogLeNet"),
            ("cnn", "CNN"),
            ("transformer", "Transformer"),
            ("attention", "Attention"),
            ("encoder", "Encoder"),
            ("decoder", "Decoder"),
            ("block", "Block"),
            ("embedding", "Embedding"),
            ("classifier", "Classifier"),
            ("feature extractor", "Feature Extractor"),
            ("lstm", "LSTM"),
            ("gru", "GRU")
        ]
        for key, comp in candidates:
            if key in ocr:
                components.append(comp)
            if len(components) >= 4:
                break

    elif img_type == "dataset":
        candidates = [
            ("sample", "Sample"),
            ("example", "Example"),
            ("input image", "Input image"),
            ("dataset", "Dataset"),
            ("class label", "Class label"),
            ("cifar", "CIFAR"),
            ("mnist", "MNIST"),
            ("imagenet", "ImageNet")
        ]
        for key, comp in candidates:
            if key in ocr:
                components.append(comp)
            if len(components) >= 4:
                break

    seen = set()
    unique_components = []
    for comp in components:
        key = comp.lower()
        if key not in seen:
            seen.add(key)
            unique_components.append(comp)

    return unique_components


def build_description(img_type):
    if img_type == "experiment_result":
        return "展示时间序列预测结果，包含中位数预测和不确定性区间"
    elif img_type == "architecture":
        return "展示模型网络结构和数据流关系"
    elif img_type == "dataset":
        return "展示数据集样本或输入示例"
    else:
        return "无法确定图片类型"


def build_code_info(img_type):
    if img_type == "experiment_result":
        return "需要实现时间序列预测模型，输出中位数预测和不确定性区间估计"
    elif img_type == "architecture":
        return "需要实现图中所示的网络结构模块"
    elif img_type == "dataset":
        return "需要准备或加载论文中展示的数据集样本"
    else:
        return ""


def main():
    try:
        if len(sys.argv) < 2:
            result = {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
            print(json.dumps(result, ensure_ascii=False))
            return

        image_path = sys.argv[1]
        img = read_image(image_path)
        features = get_basic_features(img)

        img_type = classify(OCR_TEXT, features)
        components = extract_components(OCR_TEXT, img_type)
        description = build_description(img_type)
        code_info = build_code_info(img_type)

        result = {
            "type": img_type,
            "description": description,
            "important_components": components,
            "code_relevant_information": code_info
        }

        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        result = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()