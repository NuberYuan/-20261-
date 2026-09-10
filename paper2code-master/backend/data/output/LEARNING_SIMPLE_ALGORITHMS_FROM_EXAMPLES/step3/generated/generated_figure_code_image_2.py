import sys
import json
import numpy as np
import cv2

def main():
    try:
        image_path = sys.argv[1]
        
        # 读取图片（支持中文路径）
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if img is None:
            result = {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
            print(json.dumps(result, ensure_ascii=False))
            return
        
        # OCR文字（外部提供）
        ocr_text = """@utput
Input
Tape
Gr id
Empty
Hidden
Future
state
Tant
tant
Input 195
Grid
5 55
t1
t2"""
        
        ocr_lower = ocr_text.lower()
        
        # 实验相关关键词
        experiment_keywords = [
            "accuracy", "loss", "epoch", "precision", "recall", "f1",
            "performance", "comparison", "result", "experiment",
            "training", "validation", "test", "curve", "chart",
            "准确率", "损失", "精度", "召回", "实验", "结果"
        ]
        
        # 架构相关关键词
        architecture_keywords = [
            "cnn", "transformer", "attention", "encoder", "decoder",
            "block", "embedding", "classifier", "feature extractor",
            "alexnet", "resnet", "vgg", "googlenet",
            "network", "layer", "module", "architecture",
            "input", "output", "hidden", "state", "grid"
        ]
        
        # 数据集相关关键词
        dataset_keywords = [
            "sample", "example", "dataset", "input image", "class",
            "category", "instance", "数据", "样本", "示例"
        ]
        
        # 判断类型
        has_experiment = any(kw in ocr_lower for kw in experiment_keywords)
        has_architecture = any(kw in ocr_lower for kw in architecture_keywords)
        has_dataset = any(kw in ocr_lower for kw in dataset_keywords)
        
        # 检查是否有数字/指标特征（实验结果的强信号）
        has_numbers = any(c.isdigit() for c in ocr_text)
        has_metric_pattern = any(kw in ocr_lower for kw in ["accuracy", "loss", "precision", "recall", "f1", "epoch"])
        
        # 架构特征：有Input/Output/Hidden/State等流程词汇
        has_flow_words = any(kw in ocr_lower for kw in ["input", "output", "hidden", "state", "grid", "tape"])
        
        # 优先级判断：experiment_result > architecture > dataset > other
        if has_metric_pattern or (has_experiment and has_numbers):
            result = {
                "type": "experiment_result",
                "description": "展示实验结果或性能指标对比",
                "important_components": [],
                "code_relevant_information": "需要实现模型训练和评估流程，包括指标计算"
            }
        elif has_architecture and has_flow_words:
            # 提取可能的组件名称
            components = []
            arch_terms = ["cnn", "transformer", "attention", "encoder", "decoder", 
                         "alexnet", "resnet", "vgg", "googlenet", "embedding",
                         "classifier", "feature extractor"]
            for term in arch_terms:
                if term in ocr_lower:
                    components.append(term.title())
            
            # 检查是否有Input/Output/Hidden/State等流程组件
            flow_components = []
            if "input" in ocr_lower:
                flow_components.append("Input Layer")
            if "output" in ocr_lower:
                flow_components.append("Output Layer")
            if "hidden" in ocr_lower:
                flow_components.append("Hidden State")
            if "state" in ocr_lower:
                flow_components.append("State Module")
            if "grid" in ocr_lower:
                flow_components.append("Grid Module")
            if "tape" in ocr_lower:
                flow_components.append("Tape Module")
            
            components.extend(flow_components)
            
            result = {
                "type": "architecture",
                "description": "展示模型网络结构和数据流关系",
                "important_components": components,
                "code_relevant_information": "需要实现模型架构中的各组件及数据流传递逻辑"
            }
        elif has_dataset:
            result = {
                "type": "dataset",
                "description": "展示数据样本或类别示例",
                "important_components": [],
                "code_relevant_information": "需要准备相应的数据集和预处理流程"
            }
        else:
            result = {
                "type": "other",
                "description": "无法确定图片类型",
                "important_components": [],
                "code_relevant_information": ""
            }
        
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        result = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()