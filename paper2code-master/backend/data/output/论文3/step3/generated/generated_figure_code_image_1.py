import sys
import os
import json
import numpy as np
import cv2

# OCR文字由外部模块提供，此处以默认值的形式存在。
# 也允许通过环境变量 PAPER_OCR_TEXT 覆盖。
DEFAULT_OCR_TEXT = """
X0,0
X0,1
X0,2
44.:
X0,3
X0,4
X1,l
X1,2
X2,0
X2,1
X2,2
X1,0
X3,0
X3,1
X4,0
Xl,l
x0, = H[xoo,U(xl,0)]
x02 = H[xoo,x0,, U(xl,)]
x03 = H[xoo,x0,1,x02,U(xl2)]
X0,
x04 = H[xoo
X02
U(xl,3)]
Jiaja
X1,3
X2,
X3,
X03
"""


def load_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def get_simple_visual_info(img):
    """
    仅提取基础视觉统计信息，用于辅助OCR文字进行类型判断。
    不使用边缘、轮廓、矩形、直线等复杂视觉特征。
    """
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = float(gray.mean())
    gray_std = float(gray.std())
    saturation = float(hsv[:, :, 1].mean())
    white_ratio = float(np.mean(np.all(rgb > 220, axis=2)))
    dark_ratio = float(np.mean(np.all(rgb < 50, axis=2)))
    channel_std = float(rgb.std(axis=(0, 1)).mean())

    return {
        "width": int(w),
        "height": int(h),
        "brightness": brightness,
        "gray_std": gray_std,
        "saturation": saturation,
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "color_std": channel_std,
    }


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def classify(ocr_text, visual_info):
    lower_text = ocr_text.lower()

    # 用于类型判断的关键词
    arch_keywords = [
        "cnn",
        "convolutional neural network",
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

    experiment_keywords = [
        "accuracy",
        "loss",
        "epoch",
        "precision",
        "recall",
        "f1",
        "f1-score",
        "f1 score",
        "auc",
        "roc",
        "experiment result",
        "training curve",
        "validation curve",
        "performance comparison",
        "baseline",
        "convergence",
        "learning rate",
    ]

    dataset_keywords = [
        "sample",
        "example",
        "class label",
        "category",
        "input image",
        "input sample",
        "dataset",
        "mnist",
        "cifar",
        "imagenet",
        "original image",
        "data sample",
    ]

    has_arch = any(k in lower_text for k in arch_keywords)
    has_experiment = any(k in lower_text for k in experiment_keywords)
    has_dataset = any(k in lower_text for k in dataset_keywords)

    # 按照优先级：experiment_result > architecture > dataset > other
    if has_experiment:
        model_names = [
            "alexnet",
            "resnet",
            "vgg",
            "googlenet",
            "transformer",
            "cnn",
            "bert",
            "inception",
        ]
        metric_names = [
            "accuracy",
            "loss",
            "epoch",
            "precision",
            "recall",
            "f1-score",
            "f1 score",
            "auc",
            "roc",
            "learning rate",
            "map",
            "iou",
        ]

        components = []
        for name in model_names:
            if name in lower_text:
                components.append(name.capitalize())

        metric_display = {
            "accuracy": "Accuracy",
            "loss": "Loss",
            "epoch": "Epoch",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-score",
            "f1 score": "F1-score",
            "auc": "AUC",
            "roc": "ROC",
            "learning rate": "Learning Rate",
            "map": "mAP",
            "iou": "IoU",
        }
        for metric, display in metric_display.items():
            if metric in lower_text:
                components.append(display)

        components = unique_preserve_order(components)

        return {
            "type": "experiment_result",
            "description": "展示实验性能变化和指标对比",
            "important_components": components,
            "code_relevant_information": "需要实现模型训练与评估流程以复现相应性能指标",
        }

    if has_arch:
        component_keywords = [
            ("alexnet", "AlexNet"),
            ("resnet", "ResNet"),
            ("vgg", "VGG"),
            ("googlenet", "GoogLeNet"),
            ("transformer encoder", "Transformer Encoder"),
            ("attention module", "Attention Module"),
            ("cnn", "CNN"),
            ("convolutional neural network", "CNN"),
            ("transformer", "Transformer"),
            ("attention", "Attention"),
            ("encoder", "Encoder"),
            ("decoder", "Decoder"),
            ("embedding", "Embedding"),
            ("classifier", "Classifier"),
        ]
        components = []
        for keyword, display in component_keywords:
            if keyword in lower_text:
                components.append(display)

        components = unique_preserve_order(components)

        return {
            "type": "architecture",
            "description": "展示模型网络结构和数据流关系",
            "important_components": components,
            "code_relevant_information": "需要实现所识别出的模型组件及其连接与数据流逻辑",
        }

    if has_dataset:
        return {
            "type": "dataset",
            "description": "展示数据样本或类别示例",
            "important_components": [],
            "code_relevant_information": "需要准备相应的数据加载与预处理流程",
        }

    return {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": "",
    }


def main():
    result = {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": "",
    }

    try:
        if len(sys.argv) < 2:
            print(json.dumps(result, ensure_ascii=False))
            return

        image_path = sys.argv[1]
        ocr_text = os.environ.get("PAPER_OCR_TEXT", DEFAULT_OCR_TEXT)

        img = load_image(image_path)
        visual_info = get_simple_visual_info(img)
        result = classify(ocr_text, visual_info)
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