import sys
import json
import numpy as np
import cv2
from PIL import Image
import os

def analyze_image(image_path):
    try:
        # OCR文字（外部提供）
        ocr_text = "6517 0631"
        
        # 读取图片
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
        
        # 基础视觉信息
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 简单视觉特征
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # 颜色分布
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        
        # 判断图片类型
        # OCR文字分析
        ocr_lower = ocr_text.lower()
        
        # 实验相关关键词
        experiment_keywords = [
            'accuracy', 'loss', 'epoch', 'precision', 'recall', 'f1',
            'acc', 'train', 'test', 'validation', 'performance',
            'comparison', 'result', 'curve', 'plot', 'chart'
        ]
        
        # 架构相关关键词
        architecture_keywords = [
            'cnn', 'transformer', 'attention', 'encoder', 'decoder',
            'block', 'embedding', 'classifier', 'feature extractor',
            'alexnet', 'resnet', 'vgg', 'googlenet', 'layer',
            'network', 'module', 'conv', 'pooling', 'fc'
        ]
        
        # 数据集相关关键词
        dataset_keywords = [
            'sample', 'example', 'input', 'image', 'class', 'label',
            'dataset', 'data', 'instance', 'category'
        ]
        
        # 检查OCR中的关键词
        has_experiment_kw = any(kw in ocr_lower for kw in experiment_keywords)
        has_architecture_kw = any(kw in ocr_lower for kw in architecture_keywords)
        has_dataset_kw = any(kw in ocr_lower for kw in dataset_keywords)
        
        # OCR文字主要是数字，可能表示图表数据
        has_numbers = any(c.isdigit() for c in ocr_text)
        
        # 视觉特征分析
        # 检查是否有曲线特征（通过亮度变化）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # 检查颜色多样性
        color_diversity = mean_saturation
        
        # 判断逻辑
        image_type = "other"
        description = "无法确定图片类型"
        important_components = []
        code_relevant_info = ""
        
        # 实验结果的判断
        if has_experiment_kw or (has_numbers and edge_density > 0.05 and edge_density < 0.3):
            image_type = "experiment_result"
            description = "展示实验结果和性能指标"
            important_components = []
            # 尝试提取模型名称
            for model in ['alexnet', 'resnet', 'vgg', 'googlenet', 'cnn', 'transformer']:
                if model in ocr_lower:
                    important_components.append(model.capitalize())
            # 添加指标
            for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1']:
                if metric in ocr_lower:
                    important_components.append(metric.capitalize())
            code_relevant_info = "需要实现实验评估和性能对比功能"
        
        # 架构的判断
        elif has_architecture_kw:
            image_type = "architecture"
            description = "展示模型网络结构和数据流关系"
            important_components = []
            # 提取架构组件
            for comp in ['cnn', 'transformer', 'attention', 'encoder', 'decoder', 
                        'block', 'embedding', 'classifier', 'alexnet', 'resnet', 
                        'vgg', 'googlenet']:
                if comp in ocr_lower:
                    important_components.append(comp.capitalize())
            code_relevant_info = "需要实现网络结构定义和前向传播"
        
        # 数据集的判断
        elif has_dataset_kw or (edge_density < 0.05 and color_diversity > 30):
            image_type = "dataset"
            description = "展示数据样本和输入示例"
            important_components = []
            code_relevant_info = "需要实现数据加载和预处理"
        
        # 如果OCR只有数字，可能是实验结果图表
        elif has_numbers and edge_density > 0.1:
            image_type = "experiment_result"
            description = "展示实验数据结果"
            important_components = []
            code_relevant_info = "需要实现实验结果可视化"
        
        # 默认判断为other
        else:
            image_type = "other"
            description = "无法确定图片类型"
            important_components = []
            code_relevant_info = ""
        
        return {
            "type": image_type,
            "description": description,
            "important_components": important_components,
            "code_relevant_information": code_relevant_info
        }
        
    except Exception as e:
        return {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            result = {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
        else:
            image_path = sys.argv[1]
            result = analyze_image(image_path)
        
        print(json.dumps(result, ensure_ascii=False))
    except:
        result = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
        print(json.dumps(result, ensure_ascii=False))