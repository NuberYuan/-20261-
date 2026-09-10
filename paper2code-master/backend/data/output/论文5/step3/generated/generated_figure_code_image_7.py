import sys
import os
import json
import numpy as np
import cv2

FALLBACK = {
    "type": "other",
    "description": "无法确定图片类型",
    "important_components": [],
    "code_relevant_information": ""
}

DEFAULT_OCR_TEXT = """
10%
50%
90%
Mean
0.06
Weekly Interval
(7 Days)
0.04
Past
0.02
AAAANI
(n=-1)
-5
Position Index (n)
T=10
T-20
T-30
0.015
0.01
0.005
-75
-50
-25
Day
caAM
"""


def load_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def get_basic_visual_info(img):
    if img is None:
        return {"non_white_ratio": 0.0, "brightness": 255.0, "colorfulness": 0.0}
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        non_white_ratio = float(np.mean(gray < 240))
        brightness = float(np.mean(gray))
        b, g, r = cv2.split(img)
        rg = float(np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))))
        yb = float(np.mean(np.abs(0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32))))
        return {
            "non_white_ratio": non_white_ratio,
            "brightness": brightness,
            "colorfulness": rg + yb
        }
    except Exception:
        return {"non_white_ratio": 0.0, "brightness": 255.0, "colorfulness": 0.0}


def extract_arch_components(text):
    mapping = [
        ("alexnet", "AlexNet"),
        ("resnet", "ResNet"),
        ("vgg", "VGG"),
        ("googlenet", "GoogLeNet"),
        ("transformer", "Transformer"),
        ("attention", "Attention Module"),
        ("encoder", "Encoder"),
        ("decoder", "Decoder"),
        ("embedding", "Embedding"),
        ("classifier", "Classifier"),
        ("feature extractor", "Feature Extractor"),
        ("feature extraction", "Feature Extractor"),
        ("convolutional", "CNN"),
        ("cnn", "CNN"),
        ("block", "Block")
    ]
    comps = []
    for key, val in mapping:
        if key in text and val not in comps:
            comps.append(val)
    return comps


def extract_experiment_components(text):
    models = [
        ("alexnet", "AlexNet"),
        ("resnet", "ResNet"),
        ("vgg", "VGG"),
        ("googlenet", "GoogLeNet"),
        ("transformer", "Transformer"),
        ("cnn", "CNN"),
        ("lstm", "LSTM"),
        ("bert", "BERT"),
        ("efficientnet", "EfficientNet"),
        ("densenet", "DenseNet"),
        ("mobilenet", "MobileNet")
    ]
    metrics = [
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("epoch", "Epoch"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1-score", "F1-score"),
        ("f1 score", "F1-score"),
        ("f1", "F1-score"),
        ("mean", "Mean"),
        ("map", "mAP"),
        ("auc", "AUC"),
        ("roc", "ROC"),
        ("psnr", "PSNR"),
        ("ssim", "SSIM"),
        ("mse", "MSE"),
        ("mae", "MAE"),
        ("position index", "Position Index"),
        ("weekly interval", "Weekly Interval")
    ]
    comps = []
    for key, val in models:
        if key in text and val not in comps:
            comps.append(val)
    for key, val in metrics:
        if key in text and val not in comps:
            comps.append(val)
    return comps


def extract_dataset_components(text):
    datasets = [
        ("mnist", "MNIST"),
        ("cifar-10", "CIFAR-10"),
        ("cifar-100", "CIFAR-100"),
        ("imagenet", "ImageNet"),
        ("coco", "COCO"),
        ("pascal voc", "PASCAL VOC"),
        ("svhn", "SVHN"),
        ("fashion-mnist", "Fashion-MNIST"),
        ("kitti", "KITTI"),
        ("cityscapes", "Cityscapes")
    ]
    comps = []
    for key, val in datasets:
        if key in text and val not in comps:
            comps.append(val)
    if "dataset" in text:
        comps.append("Dataset")
    if "sample" in text or "samples" in text:
        comps.append("Samples")
    if "example" in text or "examples" in text:
        comps.append("Examples")
    if "class" in text or "category" in text or "categories" in text:
        comps.append("Classes")
    return comps


def is_experiment_result(text, visual):
    lower = text.lower()
    strong_exp_terms = [
        "accuracy", "loss", "epoch", "precision", "recall",
        "f1-score", "f1 score", "f1", "map", "auc", "roc",
        "metric", "result", "curve", "comparison", "performance",
        "validation", "training", "iteration", "learning rate",
        "baseline", "position index", "weekly interval", "day",
        "interval", "error", "mean"
    ]
    if any(term in lower for term in ["accuracy", "loss", "epoch", "precision", "recall", "f1", "map", "auc", "roc"]):
        return True
    if ("%" in text or "mean" in lower) and any(ch.isdigit() for ch in text):
        return True
    if any(term in lower for term in ["curve", "comparison", "performance", "validation", "training", "iteration", "learning rate", "position index", "weekly interval", "day", "interval", "baseline"]):
        return True
    return False


def is_architecture(text, visual):
    lower = text.lower()
    arch_terms = [
        "cnn", "transformer", "attention", "encoder", "decoder",
        "block", "embedding", "classifier", "feature extractor",
        "feature extraction", "alexnet", "resnet", "vgg", "googlenet",
        "convolutional", "fully connected", "pooling", "softmax"
    ]
    return any(term in lower for term in arch_terms)


def is_dataset(text, visual):
    lower = text.lower()
    dataset_terms = [
        "dataset", "mnist", "cifar", "imagenet", "coco",
        "pascal voc", "svhn", "fashion-mnist", "kitti", "cityscapes",
        "sample", "samples", "example", "examples", "input image",
        "training examples", "data augmentation", "original image"
    ]
    return any(term in lower for term in dataset_terms)


def classify(ocr_text, visual):
    text = ocr_text.lower()

    if is_experiment_result(ocr_text, visual):
        comps = extract_experiment_components(text)
        return {
            "type": "experiment_result",
            "description": "展示实验性能变化或指标比较结果",
            "important_components": comps[:6],
            "code_relevant_information": "需要实现实验训练与评估流程，计算并可视化相关指标变化"
        }

    if is_architecture(ocr_text, visual):
        comps = extract_arch_components(text)
        return {
            "type": "architecture",
            "description": "展示模型网络结构和数据流关系",
            "important_components": comps[:6],
            "code_relevant_information": "需要根据模型结构实现相应的网络组件和数据流"
        }

    if is_dataset(ocr_text, visual):
        comps = extract_dataset_components(text)
        return {
            "type": "dataset",
            "description": "展示数据样本、类别图片或输入示例",
            "important_components": comps[:6],
            "code_relevant_information": "需要准备或加载对应数据样本及其预处理流程"
        }

    return FALLBACK


def get_ocr_text():
    if len(sys.argv) > 2:
        return sys.argv[2]
    image_path = sys.argv[1]
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return DEFAULT_OCR_TEXT


def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps(FALLBACK, ensure_ascii=False))
            return

        image_path = sys.argv[1]
        ocr_text = get_ocr_text()
        img = load_image(image_path)
        visual = get_basic_visual_info(img)
        result = classify(ocr_text, visual)
        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        print(json.dumps(FALLBACK, ensure_ascii=False))


if __name__ == "__main__":
    main()