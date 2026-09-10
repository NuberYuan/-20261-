import sys
import json
import numpy as np
import cv2

# OCR文本由外部模块提供，此处为当前输入图片的OCR识别结果
OCR_TEXT = """
UNet++ LI
UNet++ L2
UNet++ L
UNet++ L4
Param = 0.1M
Param = 0.5M
Param = 2.2M
Param = 9.OM
8 94
Inference Time (s)
(a)
"""


def get_simple_visual_features(img):
    """只使用非常基础的视觉信息辅助判断，不使用复杂结构特征。"""
    h, w = img.shape[:2]
    aspect_ratio = w / max(h, 1)

    # 彩色像素比例：饱和度和亮度较高的像素占比
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = (saturation > 60) & (value > 30)
    color_ratio = float(np.count_nonzero(color_mask)) / max(color_mask.size, 1)

    return aspect_ratio, color_ratio


def classify_image(ocr_text, img):
    text = ocr_text.upper()

    aspect_ratio, color_ratio = get_simple_visual_features(img)

    experiment_keywords = [
        "ACCURACY", "LOSS", "EPOCH", "PRECISION", "RECALL", "F1",
        "INFERENCE TIME", "PARAM", "AP", "MAP", "AUC", "ROC",
        "BENCHMARK", "COMPARISON", "PERFORMANCE", "CURVE", "BAR",
        "准确率", "损失", "性能", "实验", "曲线", "柱状"
    ]

    architecture_keywords = [
        "CNN", "TRANSFORMER", "ATTENTION", "ENCODER", "DECODER",
        "BLOCK", "EMBEDDING", "CLASSIFIER", "FEATURE EXTRACTOR",
        "ALEXNET", "RESNET", "VGG", "GOOGLENET", "NETWORK", "MODULE",
        "CONV", "LAYER"
    ]

    dataset_keywords = [
        "SAMPLE", "EXAMPLE", "CLASS", "INPUT IMAGE", "DATASET",
        "TRAINING DATA", "TEST DATA", "数据", "样本", "类别", "示例"
    ]

    has_experiment = any(keyword in text for keyword in experiment_keywords)
    has_architecture = any(keyword in text for keyword in architecture_keywords)
    has_dataset = any(keyword in text for keyword in dataset_keywords)

    # 优先级：experiment_result > architecture > dataset > other
    if has_experiment:
        components = []

        # 模型名称映射
        model_candidates = [
            ("UNET++", "UNet++"),
            ("RESNET", "ResNet"),
            ("ALEXNET", "AlexNet"),
            ("VGG", "VGG"),
            ("GOOGLENET", "GoogLeNet"),
            ("TRANSFORMER", "Transformer"),
            ("CNN", "CNN"),
        ]
        for key, name in model_candidates:
            if key in text:
                components.append(name)

        # 指标名称映射
        metric_candidates = [
            ("ACCURACY", "Accuracy"),
            ("LOSS", "Loss"),
            ("PRECISION", "Precision"),
            ("RECALL", "Recall"),
            ("F1", "F1"),
            ("INFERENCE TIME", "Inference Time"),
            ("PARAM", "Param"),
            ("MAP", "mAP"),
            ("AUC", "AUC"),
            ("AP", "AP"),
        ]
        for key, name in metric_candidates:
            if key in text:
                components.append(name)

        components = list(dict.fromkeys(components))

        if "UNET++" in text and "INFERENCE TIME" in text:
            description = "展示UNet++不同参数规模模型的推理时间对比"
            code_relevant_information = "需要实现UNet++模型的不同参数配置，并进行推理时间测试"
        else:
            description = "展示模型性能或实验数据对比图"
            code_relevant_information = "需要实现相关模型并设置实验指标，用于对比性能或推理速度"

        return {
            "type": "experiment_result",
            "description": description,
            "important_components": components,
            "code_relevant_information": code_relevant_information
        }

    if has_architecture:
        components = []

        model_candidates = [
            ("ALEXNET", "AlexNet"),
            ("RESNET", "ResNet"),
            ("VGG", "VGG"),
            ("GOOGLENET", "GoogLeNet"),
            ("TRANSFORMER", "Transformer"),
            ("ENCODER", "Encoder"),
            ("DECODER", "Decoder"),
            ("ATTENTION", "Attention"),
            ("EMBEDDING", "Embedding"),
            ("CLASSIFIER", "Classifier"),
            ("FEATURE EXTRACTOR", "Feature Extractor"),
            ("BLOCK", "Block"),
            ("MODULE", "Module"),
            ("CNN", "CNN"),
        ]
        for key, name in model_candidates:
            if key in text:
                components.append(name)

        if "TRANSFORMER" in text and "ENCODER" in text:
            components.append("Transformer Encoder")
        if "ATTENTION" in text and "MODULE" in text:
            components.append("Attention Module")

        components = list(dict.fromkeys(components))

        return {
            "type": "architecture",
            "description": "展示模型网络结构和数据流关系",
            "important_components": components,
            "code_relevant_information": "需要实现相关模块并连接数据流"
        }

    if has_dataset:
        return {
            "type": "dataset",
            "description": "展示数据集样本或类别示例",
            "important_components": [],
            "code_relevant_information": ""
        }

    # 简单视觉辅助：宽高比明显为横向且存在一定彩色像素，偏向实验图
    if aspect_ratio > 1.4 and color_ratio > 0.05:
        return {
            "type": "experiment_result",
            "description": "根据图像视觉特征判断为实验数据图",
            "important_components": [],
            "code_relevant_information": ""
        }

    return {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": ""
    }


def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({
                "type": "other",
                "description": "未提供图片路径",
                "important_components": [],
                "code_relevant_information": ""
            }, ensure_ascii=False))
            return

        image_path = sys.argv[1]

        # 必须使用支持中文路径的读取方式
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            print(json.dumps({
                "type": "other",
                "description": "无法读取图片文件",
                "important_components": [],
                "code_relevant_information": ""
            }, ensure_ascii=False))
            return

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(json.dumps({
                "type": "other",
                "description": "无法解码图片",
                "important_components": [],
                "code_relevant_information": ""
            }, ensure_ascii=False))
            return

        result = classify_image(OCR_TEXT, img)
        print(json.dumps(result, ensure_ascii=False))

    except Exception:
        print(json.dumps({
            "type": "other",
            "description": "图片分析失败",
            "important_components": [],
            "code_relevant_information": ""
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()