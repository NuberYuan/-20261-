import os
import sys
import json
import numpy as np
import cv2


def safe_stdout_encoding():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def read_image(image_path):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def get_visual_hints(img):
    if img is None:
        return {}
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        non_white_mask = np.any(rgb < 240, axis=2)
        non_white_ratio = float(np.mean(non_white_mask))
        img_float = rgb.astype(np.float32)
        saturation = img_float.max(axis=2) - img_float.min(axis=2)
        mean_sat = float(np.mean(saturation))
        return {"non_white_ratio": non_white_ratio, "mean_sat": mean_sat}
    except Exception:
        return {}


def extract_components(ocr_text, candidates, max_items=8):
    if not ocr_text or not candidates:
        return []

    lines = [line.strip() for line in ocr_text.replace('\r', '\n').split('\n') if line.strip()]
    tokens = []
    for line in lines:
        for tok in line.split():
            tok = tok.strip(' ,;:()[]{}')
            if tok and tok not in tokens:
                tokens.append(tok)

    found = []
    lower_text = ocr_text.lower()

    for cand in candidates:
        if " " not in cand:
            continue
        cand_lower = cand.lower()
        idx = 0
        while True:
            pos = lower_text.find(cand_lower, idx)
            if pos == -1:
                break
            phrase = ocr_text[pos:pos + len(cand)].strip()
            if phrase and phrase not in found:
                found.append(phrase)
            idx = pos + 1

    for tok in tokens:
        low = tok.lower()
        for cand in candidates:
            if " " in cand:
                continue
            if low == cand.lower():
                if tok not in found:
                    found.append(tok)
                break

    unique = []
    for item in found:
        if item not in unique:
            unique.append(item)
        if len(unique) >= max_items:
            break
    return unique


def classify(ocr_text, hints):
    ocr_lower = ocr_text.lower() if ocr_text else ""

    experiment_keywords = [
        "accuracy", "loss", "epoch", "precision", "recall", "f1",
        "iou", "dice", "psnr", "ssim", "performance", "comparison",
        "curve", "bar chart", "learning rate", "validation"
    ]
    architecture_keywords = [
        "cnn", "transformer", "attention", "encoder", "decoder",
        "block", "embedding", "classifier", "feature extractor",
        "alexnet", "resnet", "vgg", "googlenet", "u-net", "unet++",
        "unett+", "unet"
    ]
    dataset_keywords = [
        "sample", "example", "input image", "ground truth",
        "training data", "test data", "class", "category",
        "dataset", "image", "annotation", "mask"
    ]

    is_experiment = any(k in ocr_lower for k in experiment_keywords)
    is_architecture = any(k in ocr_lower for k in architecture_keywords)
    is_dataset = any(k in ocr_lower for k in dataset_keywords)

    model_tokens = [
        "alexnet", "resnet", "vgg", "googlenet", "u-net",
        "unet++", "unett+", "wide u-net", "unet", "transformer",
        "vit", "bert"
    ]
    model_hit_count = 0
    for m in model_tokens:
        if m in ocr_lower:
            model_hit_count += 1

    if not is_experiment and model_hit_count >= 2 and ("ground truth" in ocr_lower or "image" in ocr_lower):
        is_experiment = True

    if is_experiment:
        model_candidates = [
            "alexnet", "resnet", "vgg", "googlenet", "u-net",
            "unet++", "unett+", "wide u-net", "unet", "transformer",
            "cnn", "attention", "vit", "bert"
        ]
        metric_candidates = [
            "accuracy", "loss", "epoch", "precision", "recall",
            "f1", "iou", "dice", "psnr", "ssim"
        ]
        comps = []
        comps += extract_components(ocr_text, model_candidates, 4)
        comps += extract_components(ocr_text, metric_candidates, 4)

        unique = []
        for c in comps:
            if c not in unique:
                unique.append(c)

        return {
            "type": "experiment_result",
            "description": "展示实验结果对比或性能指标变化",
            "important_components": unique[:8],
            "code_relevant_information": "需要实现模型推理并评估相关性能指标"
        }

    if is_architecture:
        arch_candidates = [
            "cnn", "transformer", "attention", "encoder", "decoder",
            "block", "embedding", "classifier", "feature extractor",
            "alexnet", "resnet", "vgg", "googlenet", "u-net",
            "unet++", "unett+", "wide u-net", "unet"
        ]
        comps = extract_components(ocr_text, arch_candidates, 8)
        return {
            "type": "architecture",
            "description": "展示模型网络结构和数据流关系",
            "important_components": comps,
            "code_relevant_information": "需要实现相应的网络结构模块"
        }

    if is_dataset:
        return {
            "type": "dataset",
            "description": "展示数据集样本或输入示例",
            "important_components": [],
            "code_relevant_information": "需要准备或加载数据集并进行预处理"
        }

    if not ocr_lower.strip():
        non_white_ratio = hints.get("non_white_ratio", 0.0)
        mean_sat = hints.get("mean_sat", 0.0)
        if non_white_ratio > 0.05 and mean_sat > 15.0:
            return {
                "type": "dataset",
                "description": "根据视觉特征推测为数据样本",
                "important_components": [],
                "code_relevant_information": "需要准备或加载数据集并进行预处理"
            }

    return {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": ""
    }


def emit_json(obj):
    try:
        print(json.dumps(obj, ensure_ascii=False))
    except Exception:
        print('{"type":"other","description":"无法确定图片类型","important_components":[],"code_relevant_information":""}')


def main():
    safe_stdout_encoding()

    fallback = {
        "type": "other",
        "description": "无法确定图片类型",
        "important_components": [],
        "code_relevant_information": ""
    }

    try:
        if len(sys.argv) < 2:
            emit_json(fallback)
            return

        image_path = sys.argv[1]
        ocr_text = sys.argv[2] if len(sys.argv) > 2 else ""

        img = read_image(image_path)
        hints = get_visual_hints(img)
        result = classify(ocr_text, hints)
        emit_json(result)
    except Exception:
        emit_json(fallback)


if __name__ == "__main__":
    main()