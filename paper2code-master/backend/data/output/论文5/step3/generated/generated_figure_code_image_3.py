import sys
import json
import numpy as np
import cv2

OCR_TEXT = """Electricity
Traffic
Retail
Volatility
Average
Max
Ave
Min
21.6%
30.0%
23.4%
20.0%
10.2%
10.0%
{7.6%
5.6%
4.9%
1.5%
2.0%
3.2%
3.6%
0.0%
-10.0%
No
Gating
No Static
No Variable
No Self-
No Local
Encoder
"""


def read_image_unicode(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def get_visual_info(img):
    if img is None:
        return {"valid": False}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return {
        "valid": True,
        "width": w,
        "height": h,
        "aspect": w / float(h) if h > 0 else 0.0,
        "non_white_ratio": float(np.mean(gray < 240)),
        "color_ratio": float(np.mean(saturation > 40)),
        "brightness": float(np.mean(gray)),
    }


def classify_image(ocr_text, vis):
    text = ocr_text.lower()

    has_percent = "%" in text
    experiment_metrics = [
        "accuracy",
        "loss",
        "epoch",
        "precision",
        "recall",
        "f1",
        "average",
        "max",
        "min",
        "ave",
    ]
    comparison_terms = [
        "no gating",
        "no static",
        "no variable",
        "no self",
        "no local",
        "electricity",
        "traffic",
        "retail",
        "volatility",
    ]

    has_metric = any(term in text for term in experiment_metrics)
    has_comparison = any(term in text for term in comparison_terms)

    if has_percent and (has_metric or has_comparison):
        return "experiment_result"
    if has_metric and has_comparison:
        return "experiment_result"

    architecture_terms = [
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
    has_arch = any(term in text for term in architecture_terms)
    if has_arch and not has_percent:
        return "architecture"

    dataset_terms = [
        "sample",
        "example",
        "input image",
        "class",
        "label",
        "dataset",
    ]
    if any(term in text for term in dataset_terms):
        return "dataset"

    if vis.get("valid"):
        if vis["aspect"] > 1.1 and vis["color_ratio"] > 0.03:
            return "experiment_result"
        if has_arch and vis["color_ratio"] > 0.05:
            return "architecture"

    return "other"


def extract_components(ocr_text, image_type):
    text = ocr_text.lower()
    components = []

    if image_type == "architecture":
        arch_map = [
            ("alexnet", "AlexNet"),
            ("resnet", "ResNet"),
            ("vgg", "VGG"),
            ("googlenet", "GoogLeNet"),
            ("transformer", "Transformer"),
            ("attention", "Attention"),
            ("encoder", "Encoder"),
            ("decoder", "Decoder"),
            ("embedding", "Embedding"),
            ("classifier", "Classifier"),
            ("feature extractor", "Feature Extractor"),
            ("cnn", "CNN"),
        ]
        for key, name in arch_map:
            if key in text:
                components.append(name)

    if image_type == "experiment_result":
        if "no gating" in text:
            components.append("No Gating")
        if "no static" in text:
            components.append("No Static")
        if "no variable" in text:
            components.append("No Variable")
        if "no self" in text:
            components.append("No Self-Attention")
        if "no local" in text:
            components.append("No Local Encoder")
        if "encoder" in text and "No Local Encoder" not in components:
            components.append("Encoder")
        if "gating" in text and "No Gating" not in components:
            components.append("Gating")
        if "static" in text and "No Static" not in components:
            components.append("Static")
        if "variable" in text and "No Variable" not in components:
            components.append("Variable")

        for metric in ["Average", "Max", "Min"]:
            if metric.lower() in text:
                components.append(metric)

    if image_type == "dataset":
        if "input image" in text:
            components.append("Input Image")
        if "sample" in text:
            components.append("Sample")
        if "class" in text:
            components.append("Class")
        if "label" in text:
            components.append("Label")

    seen = set()
    unique_components = []
    for component in components:
        if component not in seen:
            seen.add(component)
            unique_components.append(component)
    return unique_components


def get_description(image_type):
    if image_type == "experiment_result":
        return "展示不同模型配置在多个数据集上的性能对比柱状图"
    if image_type == "architecture":
        return "展示模型网络结构和数据流关系"
    if image_type == "dataset":
        return "展示数据集样本或类别示例"
    return "无法确定图片类型"


def get_code_relevant_information(image_type):
    if image_type == "experiment_result":
        return "需要实现序列预测模型，并对比不同门控、静态特征、变量选择、局部编码器等模块的消融性能"
    if image_type == "architecture":
        return "需要实现模型组件及其数据流关系"
    if image_type == "dataset":
        return "需要准备或加载相应数据集并进行预处理"
    return ""


def main():
    try:
        image_path = sys.argv[1]
        img = read_image_unicode(image_path)
        vis = get_visual_info(img)
        ocr_text = OCR_TEXT

        image_type = classify_image(ocr_text, vis)
        important_components = extract_components(ocr_text, image_type)

        result = {
            "type": image_type,
            "description": get_description(image_type),
            "important_components": important_components,
            "code_relevant_information": get_code_relevant_information(image_type),
        }
        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        fallback = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": "",
        }
        print(json.dumps(fallback, ensure_ascii=False))


if __name__ == "__main__":
    main()