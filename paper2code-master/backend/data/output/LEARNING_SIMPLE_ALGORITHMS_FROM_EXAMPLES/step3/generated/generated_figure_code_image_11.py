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
        
        # OCR文字信息（外部提供，此处为空）
        ocr_text = ""
        
        # 论文关键词（用于辅助判断）
        architecture_keywords = [
            "CNN", "Transformer", "Attention", "Encoder", "Decoder",
            "Block", "Embedding", "Classifier", "Feature Extractor",
            "AlexNet", "ResNet", "VGG", "GoogLeNet", "Network",
            "Layer", "Module", "Architecture", "Model Structure"
        ]
        
        experiment_keywords = [
            "Accuracy", "Loss", "Epoch", "Precision", "Recall",
            "F1", "Performance", "Comparison", "Result", "Experiment",
            "Training", "Validation", "Test", "Curve", "Chart",
            "Bar", "Plot", "Figure"
        ]
        
        dataset_keywords = [
            "Sample", "Example", "Input", "Dataset", "Class",
            "Category", "Image", "Data"
        ]
        
        # 简单视觉信息分析
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 分析图片颜色分布
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 检测是否包含大量彩色区域（可能是图表）
        saturation = hsv[:, :, 1]
        color_ratio = np.sum(saturation > 50) / (height * width)
        
        # 检测是否包含大量白色背景（可能是架构图）
        white_ratio = np.sum(gray > 200) / (height * width)
        
        # 检测是否包含大量文字（可能是图表或架构图）
        edges = cv2.Canny(gray, 50, 150)
        text_density = np.sum(edges > 0) / (height * width)
        
        # 判断逻辑
        ocr_lower = ocr_text.lower()
        
        # 检查OCR中的关键词
        has_architecture_kw = any(kw.lower() in ocr_lower for kw in architecture_keywords)
        has_experiment_kw = any(kw.lower() in ocr_lower for kw in experiment_keywords)
        has_dataset_kw = any(kw.lower() in ocr_lower for kw in dataset_keywords)
        
        # 提取重要组件
        important_components = []
        
        # 从OCR中提取模型名称
        model_names = ["AlexNet", "ResNet", "VGG", "GoogLeNet", "CNN", 
                       "Transformer", "Attention", "Encoder", "Decoder",
                       "LSTM", "RNN", "BERT", "GPT"]
        
        for model in model_names:
            if model.lower() in ocr_lower:
                important_components.append(model)
        
        # 从OCR中提取指标名称
        metric_names = ["Accuracy", "Loss", "Precision", "Recall", "F1", 
                        "F1-score", "Epoch", "MSE", "RMSE", "MAE"]
        
        for metric in metric_names:
            if metric.lower() in ocr_lower:
                important_components.append(metric)
        
        # 去重
        important_components = list(set(important_components))
        
        # 判断类型
        img_type = "other"
        description = "无法确定图片类型"
        code_info = ""
        
        # 优先级：experiment_result > architecture > dataset > other
        
        # 判断是否为experiment_result
        if has_experiment_kw or (color_ratio > 0.1 and text_density > 0.05):
            # 检查是否包含实验相关特征
            if has_experiment_kw or any(kw.lower() in ocr_lower for kw in 
                                       ["accuracy", "loss", "epoch", "precision", 
                                        "recall", "f1", "performance", "comparison"]):
                img_type = "experiment_result"
                description = "展示实验结果和性能对比"
                code_info = "需要实现模型训练和评估流程，包括指标计算"
                
                # 如果没有提取到组件，尝试从视觉特征判断
                if not important_components:
                    if color_ratio > 0.15:
                        important_components = ["Performance Metrics"]
        
        # 判断是否为architecture
        if img_type == "other" and (has_architecture_kw or 
                                    (white_ratio > 0.5 and text_density > 0.03)):
            if has_architecture_kw or any(kw.lower() in ocr_lower for kw in 
                                         ["network", "layer", "module", "architecture",
                                          "model structure", "block", "embedding"]):
                img_type = "architecture"
                description = "展示模型网络结构和数据流关系"
                code_info = "需要实现模型架构，包括各层定义和前向传播"
                
                # 如果没有提取到组件，尝试从视觉特征判断
                if not important_components:
                    if white_ratio > 0.6 and text_density > 0.04:
                        important_components = ["Network Architecture"]
        
        # 判断是否为dataset
        if img_type == "other" and has_dataset_kw:
            if any(kw.lower() in ocr_lower for kw in 
                   ["sample", "example", "input", "dataset", "class", "category"]):
                img_type = "dataset"
                description = "展示数据集样本和类别示例"
                code_info = "需要准备数据加载和预处理模块"
                
                if not important_components:
                    important_components = ["Dataset Samples"]
        
        # 如果仍然无法判断，使用视觉特征
        if img_type == "other":
            # 根据视觉特征进行简单判断
            if color_ratio > 0.2 and text_density > 0.08:
                img_type = "experiment_result"
                description = "展示实验结果图表"
                code_info = "需要实现实验评估和结果可视化"
                important_components = ["Experimental Results"]
            elif white_ratio > 0.7 and text_density > 0.05:
                img_type = "architecture"
                description = "展示模型架构图"
                code_info = "需要实现模型架构定义"
                important_components = ["Model Architecture"]
            elif color_ratio > 0.15 and text_density < 0.03:
                img_type = "dataset"
                description = "展示数据集样本"
                code_info = "需要准备数据集和预处理"
                important_components = ["Dataset"]
        
        # 构建结果
        result = {
            "type": img_type,
            "description": description,
            "important_components": important_components,
            "code_relevant_information": code_info
        }
        
        # 输出JSON
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        # 任何异常都输出默认JSON
        result = {
            "type": "other",
            "description": "无法确定图片类型",
            "important_components": [],
            "code_relevant_information": ""
        }
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()