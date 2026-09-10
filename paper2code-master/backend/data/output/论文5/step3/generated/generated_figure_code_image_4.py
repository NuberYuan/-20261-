import sys
import os
import json
import numpy as np
import cv2


def get_ocr_text(image_path):
    if len(sys.argv) > 2:
        return sys.argv[2]
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            pass
    return ''


def format_component(name):
    mapping = {
        "alexnet": "AlexNet",
        "resnet": "ResNet",
        "vgg": "VGG",
        "googlenet": "GoogLeNet",
        "cnn": "CNN",
        "transformer": "Transformer",
        "attention": "Attention",
        "encoder": "Encoder",
        "decoder": "Decoder",
        "block": "Block",
        "embedding": "Embedding",
        "classifier": "Classifier",
        "feature extractor": "Feature Extractor",
        "accuracy": "Accuracy",
        "loss": "Loss",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "f1-score": "F1-score",
        "average": "Average",
        "max": "Max",
        "min": "Min",
        "volatility": "Volatility"
    }
    return mapping.get(name, name)


def extract_components(ocr_lower, experiment_mode):
    model_names = [
        "alexnet", "resnet", "vgg", "googlenet", "cnn",
        "transformer", "attention", "encoder", "decoder", "block",
        "embedding", "classifier", "feature extractor"
    ]
    metric_names = [
        "accuracy", "loss", "precision", "recall", "f1",
        "f1-score", "average", "max", "min", "volatility"
    ]
    components = []
    if experiment_mode:
        for m in model_names:
            if m in ocr_lower:
                components.append(format_component(m))
        for m in metric_names:
            if m in ocr_lower:
                components.append(format_component(m))
    else:
        for m in model_names:
            if m in ocr_lower:
                components.append(format_component(m))
    seen = set()
    unique = []
    for c in components:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique[:5]


def analyze_image(image_path, ocr_text=''):
    result = {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": ""
    }
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return result

        h, w = img.shape[:2]
        aspect_ratio = w / h if h else 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_sat = float(np.mean(saturation))
        sat_std = float(np.std(saturation))

        ocr_lower = ocr_text.lower()

        arch_keywords = [
            "cnn", "transformer", "attention", "encoder", "decoder",
            "block", "embedding", "classifier", "feature extractor",
            "alexnet", "resnet", "vgg", "googlenet",
            "convolution", "layer", "network", "module",
            "softmax", "relu", "pooling"
        ]
        exp_keywords = [
            "accuracy", "loss", "epoch", "precision", "recall",
            "f1", "f1-score", "average", "max", "min",
            "volatility", "%", "baseline", "proposed",
            "ablation", "experiment", "result", "performance"
        ]
        dataset_keywords = [
            "sample", "example", "dataset", "images", "class",
            "label", "training data", "test data",
            "cifar", "mnist", "imagenet"
        ]

        def count_hits(kw_list):
            return sum(1 for kw in kw_list if kw in ocr_lower)

        arch_hits = count_hits(arch_keywords)
        exp_hits = count_hits(exp_keywords)
        dataset_hits = count_hits(dataset_keywords)

        if exp_hits > 0:
            result["type"] = "experiment_result"
            result["description"] = "展示实验性能对比结果"
            result["important_components"] = extract_components(ocr_lower, experiment_mode=True)
            result["code_relevant_information"] = "需要实现性能评估指标计算和对比实验"
        elif arch_hits > 0 and dataset_hits == 0:
            result["type"] = "architecture"
            result["description"] = "展示模型网络结构和数据流关系"
            result["important_components"] = extract_components(ocr_lower, experiment_mode=False)
            result["code_relevant_information"] = "需要实现相应的神经网络模块"
        elif dataset_hits > 0:
            result["type"] = "dataset"
            result["description"] = "展示数据集样本或类别图片"
            result["code_relevant_information"] = "需要准备对应数据集或数据加载器"
        else:
            if avg_sat > 50 and sat_std > 40:
                result["type"] = "experiment_result"
                result["description"] = "根据图像色彩特征判断为实验性能图"
                result["code_relevant_information"] = "需要实现性能评估"
            elif aspect_ratio > 2.2:
                result["type"] = "architecture"
                result["description"] = "根据图像宽高比判断为模型结构图"
                result["code_relevant_information"] = "需要实现网络结构"
    except Exception:
        result = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
    return result


if __name__ == "__main__":
    try:
        image_path = sys.argv[1]
        ocr_text = get_ocr_text(image_path)
        result = analyze_image(image_path, ocr_text)
        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        print(json.dumps({
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }, ensure_ascii=False))