import sys
import json
import numpy as np
import cv2
from PIL import Image
import os

def analyze_image(image_path):
    try:
        # Read image using numpy fromfile to support Chinese paths
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
        
        # OCR text provided externally
        ocr_text = "output\nTape\nInput"
        
        # Convert to lowercase for easier matching
        ocr_lower = ocr_text.lower()
        
        # Define keywords for each type
        architecture_keywords = [
            'cnn', 'transformer', 'attention', 'encoder', 'decoder',
            'block', 'embedding', 'classifier', 'feature extractor',
            'alexnet', 'resnet', 'vgg', 'googlenet', 'network',
            'layer', 'convolution', 'pooling', 'fully connected',
            'softmax', 'activation', 'module', 'architecture'
        ]
        
        experiment_keywords = [
            'accuracy', 'loss', 'epoch', 'precision', 'recall',
            'f1', 'performance', 'comparison', 'result', 'experiment',
            'training', 'validation', 'test', 'curve', 'plot',
            'chart', 'graph', 'baseline', 'improvement', 'error rate',
            'convergence', 'iteration', 'batch'
        ]
        
        dataset_keywords = [
            'sample', 'example', 'input image', 'dataset', 'class',
            'category', 'label', 'instance', 'training data',
            'test data', 'ground truth', 'annotation'
        ]
        
        # Check for experiment_result keywords first (highest priority)
        experiment_found = []
        for keyword in experiment_keywords:
            if keyword in ocr_lower:
                experiment_found.append(keyword)
        
        # Check for architecture keywords
        architecture_found = []
        for keyword in architecture_keywords:
            if keyword in ocr_lower:
                architecture_found.append(keyword)
        
        # Check for dataset keywords
        dataset_found = []
        for keyword in dataset_keywords:
            if keyword in ocr_lower:
                dataset_found.append(keyword)
        
        # Simple visual analysis - check if image has multiple distinct regions
        # (suggesting a diagram or architecture)
        h, w = img.shape[:2]
        
        # Convert to grayscale for simple analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check color variance - experiment results often have colorful plots
        b, g, r = cv2.split(img)
        color_variance = np.std(b) + np.std(g) + np.std(r)
        
        # Check if image has text-like structures (using simple threshold)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_density = np.sum(binary > 0) / (h * w)
        
        # Decision logic with priority: experiment_result > architecture > dataset > other
        if experiment_found:
            # Likely experiment result
            components = []
            # Extract model names if present
            for model in ['alexnet', 'resnet', 'vgg', 'googlenet', 'cnn', 'transformer']:
                if model in ocr_lower:
                    components.append(model.capitalize() if model != 'cnn' else 'CNN')
            
            # Extract metric names
            for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1']:
                if metric in ocr_lower:
                    components.append(metric.capitalize() if metric != 'f1' else 'F1-score')
            
            if not components:
                components = []
            
            return {
                "type": "experiment_result",
                "description": "展示实验结果和性能指标对比",
                "important_components": components,
                "code_relevant_information": "需要实现模型训练和评估流程，包括性能指标计算"
            }
        
        elif architecture_found and len(architecture_found) >= 2:
            # Likely architecture diagram
            components = []
            # Extract specific architecture components
            for comp in ['cnn', 'transformer', 'attention', 'encoder', 'decoder', 
                        'alexnet', 'resnet', 'vgg', 'googlenet', 'embedding', 'classifier']:
                if comp in ocr_lower:
                    if comp == 'cnn':
                        components.append('CNN')
                    elif comp == 'alexnet':
                        components.append('AlexNet')
                    elif comp == 'resnet':
                        components.append('ResNet')
                    elif comp == 'vgg':
                        components.append('VGG')
                    elif comp == 'googlenet':
                        components.append('GoogLeNet')
                    elif comp == 'transformer':
                        components.append('Transformer')
                    elif comp == 'attention':
                        components.append('Attention Module')
                    elif comp == 'encoder':
                        components.append('Encoder')
                    elif comp == 'decoder':
                        components.append('Decoder')
                    elif comp == 'embedding':
                        components.append('Embedding Layer')
                    elif comp == 'classifier':
                        components.append('Classifier')
            
            return {
                "type": "architecture",
                "description": "展示模型网络结构和数据流关系",
                "important_components": components,
                "code_relevant_information": "需要实现模型架构中的各个组件和模块"
            }
        
        elif dataset_found:
            # Likely dataset example
            return {
                "type": "dataset",
                "description": "展示数据样本和输入示例",
                "important_components": [],
                "code_relevant_information": "需要准备相应的数据集和预处理流程"
            }
        
        else:
            # Check if image might be architecture based on visual complexity
            # Simple heuristic: architecture diagrams often have moderate text density
            # and structured layout
            if text_density > 0.05 and text_density < 0.3 and color_variance < 300:
                return {
                    "type": "architecture",
                    "description": "展示模型网络结构和数据流关系",
                    "important_components": [],
                    "code_relevant_information": "需要实现模型架构中的各个组件和模块"
                }
            else:
                return {
                    "type": "other",
                    "description": "无法确定图片类型",
                    "important_components": [],
                    "code_relevant_information": ""
                }
    
    except Exception as e:
        # Catch all exceptions and return default
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
        
        # Output only JSON
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        # Final fallback
        fallback = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
        print(json.dumps(fallback, ensure_ascii=False))