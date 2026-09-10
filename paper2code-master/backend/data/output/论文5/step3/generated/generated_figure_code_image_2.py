import os
import sys
import json

import numpy as np
import cv2


def read_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def compute_visual(img):
    if img is None:
        return {
            "ink_ratio": 0.0,
            "white_ratio": 1.0,
            "color_ratio": 0.0,
            "mean_saturation": 0.0,
            "mean_value": 255.0,
            "height": 0,
            "width": 0,
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    ink_ratio = float(np.mean(gray < 160))
    white_ratio = float(np.mean(gray > 230))
    color_ratio = float(np.mean(saturation > 50))
    mean_saturation = float(np.mean(saturation))
    mean_value = float(np.mean(value))

    return {
        "ink_ratio": ink_ratio,
        "white_ratio": white_ratio,
        "color_ratio": color_ratio,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "height": h,
        "width": w,
    }


def load_ocr_text(argv):
    if len(argv) > 2:
        candidate = argv[2]
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                pass
        return candidate

    image_path = argv[1]
    base, _ = os.path.splitext(image_path)
    directory = os.path.dirname(image_path) or "."

    candidates = [
        base + ".txt",
        base + "_ocr.txt",
        os.path.join(directory, "ocr.txt"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                continue

    return ""


def contains_any(text, keywords):
    lowered = text.lower()
    for keyword in keywords:
        if keyword in lowered:
            return True
    return False


def is_likely_dataset_visual(visual):
    if not visual:
        return False

    if visual.get("color_ratio", 0.0) > 0.12 and visual.get("ink_ratio", 0.0) > 0.25:
        return True

    if visual.get("mean_saturation", 0.0) > 45.0 and visual.get("ink_ratio", 0.0) > 0.22:
        return True

    return False


def build_other():
    return {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": "",
    }


def get_architecture_components(lower_text):
    candidates = [
        ("masked interpretable multi-head attention", "Masked Interpretable Multi-head Attention"),
        ("gated residual network", "Gated Residual Network"),
        ("multi-head attention", "Multi-head Attention"),
        ("self-attention", "Self-Attention"),
        ("temporal fusion decoder", "Temporal Fusion Decoder"),
        ("feature extractor", "Feature Extractor"),
        ("convolutional neural network", "Convolutional Neural Network"),
        ("fully connected", "Fully Connected"),
        ("position-wise", "Position-wise"),
        ("feed-forward", "Feed-forward"),
        ("feedforward", "Feed-forward"),
        ("variable selection", "Variable Selection"),
        ("external context", "External Context"),
        ("primary context", "Primary Context"),
        ("add & norm", "Add & Norm"),
        ("layer norm", "Layer Norm"),
        ("batch norm", "Batch Norm"),
        ("alexnet", "AlexNet"),
        ("googlenet", "GoogLeNet"),
        ("resnet", "ResNet"),
        ("vgg", "VGG"),
        ("transformer", "Transformer"),
        ("encoder", "Encoder"),
        ("decoder", "Decoder"),
        ("attention", "Attention"),
        ("embedding", "Embedding"),
        ("classifier", "Classifier"),
        ("residual", "Residual"),
        ("dropout", "Dropout"),
        ("dense", "Dense"),
        ("gate", "Gate"),
        ("grn", "GRN"),
        ("cnn", "CNN"),
        ("rnn", "RNN"),
        ("lstm", "LSTM"),
        ("gru", "GRU"),
        ("bert", "BERT"),
        ("gan", "GAN"),
        ("mlp", "MLP"),
        ("convolution", "Convolution"),
        ("pooling", "Pooling"),
        ("backbone", "Backbone"),
        ("elu", "ELU"),
        ("relu", "ReLU"),
        ("softmax", "Softmax"),
    ]

    comps = []
    for key, name in candidates:
        if key in lower_text and name not in comps:
            comps.append(name)

    filtered = []
    for name in comps:
        if name == "Attention" and (
            "Self-Attention" in comps
            or "Multi-head Attention" in comps
            or "Masked Interpretable Multi-head Attention" in comps
        ):
            continue
        if name == "Decoder" and "Temporal Fusion Decoder" in comps:
            continue
        if name == "GRN" and "Gated Residual Network" in comps:
            continue
        if name == "CNN" and "Convolutional Neural Network" in comps:
            continue
        if name not in filtered:
            filtered.append(name)

    return filtered


def get_experiment_components(lower_text):
    model_candidates = [
        ("alexnet", "AlexNet"),
        ("googlenet", "GoogLeNet"),
        ("resnet", "ResNet"),
        ("vgg", "VGG"),
        ("transformer", "Transformer"),
        ("cnn", "CNN"),
        ("rnn", "RNN"),
        ("lstm", "LSTM"),
        ("gru", "GRU"),
        ("bert", "BERT"),
        ("gan", "GAN"),
        ("mlp", "MLP"),
    ]

    metric_candidates = [
        ("f1-score", "F1-score"),
        ("f1 score", "F1-score"),
        ("f1", "F1-score"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("epoch", "Epoch"),
        ("auc", "AUC"),
        ("roc", "ROC"),
        ("bleu", "BLEU"),
        ("rouge", "ROUGE"),
        ("mse", "MSE"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("map", "mAP"),
        ("learning rate", "Learning Rate"),
        ("error rate", "Error Rate"),
    ]

    comps = []
    for key, name in model_candidates + metric_candidates:
        if key in lower_text and name not in comps:
            comps.append(name)

    return comps


def get_dataset_components(lower_text):
    candidates = [
        ("mnist", "MNIST"),
        ("cifar-10", "CIFAR-10"),
        ("cifar-100", "CIFAR-100"),
        ("cifar", "CIFAR"),
        ("imagenet", "ImageNet"),
        ("coco", "COCO"),
        ("pascal voc", "PASCAL VOC"),
        ("cityscapes", "Cityscapes"),
        ("kitti", "KITTI"),
        ("fashion-mnist", "Fashion-MNIST"),
        ("svhn", "SVHN"),
    ]

    comps = []
    for key, name in candidates:
        if key in lower_text and name not in comps:
            comps.append(name)

    return comps


def classify(ocr_text, visual):
    lower = ocr_text.lower()

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
        "bleu",
        "rouge",
        "validation",
        "training loss",
        "test accuracy",
        "result",
        "performance",
        "comparison",
        "ablation",
        "baseline",
        "state-of-the-art",
        "sota",
        "error rate",
        "convergence",
        "learning rate",
        "batch size",
        "experiment",
        "plot",
        "curve",
        "bar chart",
        "line chart",
        "table",
        "versus",
        " vs.",
        "vs.",
    ]

    architecture_keywords = [
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
        "neural network",
        "network architecture",
        "layer",
        "convolution",
        "pooling",
        "fully connected",
        "feed-forward",
        "self-attention",
        "multi-head attention",
        "residual",
        "dropout",
        "gate",
        "dense",
        "gated residual network",
        "grn",
        "temporal fusion decoder",
        "position-wise",
        "input",
        "output",
        "backbone",
        "head",
        "module",
        "conv",
        "relu",
        "elu",
        "softmax",
        "batch norm",
        "layer norm",
        "add & norm",
        "classification",
        "convolutional",
        "deconvolution",
        "upsampling",
        "downsampling",
    ]

    dataset_keywords = [
        "dataset",
        "data sample",
        "samples",
        "examples",
        "input example",
        "training images",
        "ground truth",
        "original image",
        "class label",
        "sample images",
        "instance",
        "mnist",
        "cifar",
        "imagenet",
        "coco",
        "pascal voc",
        "cityscapes",
        "kitti",
    ]

    has_experiment = contains_any(lower, experiment_keywords)
    has_architecture = contains_any(lower, architecture_keywords)
    has_dataset = contains_any(lower, dataset_keywords)

    if has_experiment:
        comps = get_experiment_components(lower)
        description = "展示实验性能曲线或指标对比结果"
        if comps:
            code_relevant = "需要复现相关实验并输出指标：" + "、".join(comps[:10])
        else:
            code_relevant = "需要复现实验设置并绘制性能曲线或对比图表"
        return {
            "type": "experiment_result",
            "description": description,
            "important_components": comps[:12],
            "code_relevant_information": code_relevant,
        }

    if has_architecture:
        comps = get_architecture_components(lower)
        description = "展示模型网络结构和数据流关系"
        if comps:
            code_relevant = "需要实现以下组件：" + "、".join(comps[:12])
        else:
            code_relevant = "需要实现图中展示的模型网络结构"
        return {
            "type": "architecture",
            "description": description,
            "important_components": comps[:16],
            "code_relevant_information": code_relevant,
        }

    if has_dataset or is_likely_dataset_visual(visual):
        comps = get_dataset_components(lower)
        description = "展示数据样本或类别示例"
        code_relevant = "需要加载并预处理图中对应的数据集"
        return {
            "type": "dataset",
            "description": description,
            "important_components": comps[:12],
            "code_relevant_information": code_relevant,
        }

    return build_other()


def main():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        if len(sys.argv) < 2:
            print(json.dumps(build_other(), ensure_ascii=False))
            return

        image_path = sys.argv[1]
        ocr_text = load_ocr_text(sys.argv)

        visual = {
            "ink_ratio": 0.0,
            "white_ratio": 1.0,
            "color_ratio": 0.0,
            "mean_saturation": 0.0,
            "mean_value": 255.0,
            "height": 0,
            "width": 0,
        }

        try:
            img = read_image(image_path)
            if img is not None:
                visual = compute_visual(img)
        except Exception:
            pass

        result = classify(ocr_text, visual)
        print(json.dumps(result, ensure_ascii=False))

    except Exception:
        print(json.dumps(build_other(), ensure_ascii=False))


if __name__ == "__main__":
    main()