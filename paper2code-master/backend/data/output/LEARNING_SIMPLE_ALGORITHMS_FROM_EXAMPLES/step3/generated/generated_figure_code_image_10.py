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
        
        # 检测是否包含大量彩色区域（实验图表通常有彩色曲线/柱状图）
        saturation = hsv[:, :, 1]
        color_ratio = np.sum(saturation > 50) / (height * width)
        
        # 检测是否包含大量白色背景（架构图通常白底黑字）
        white_ratio = np.sum(gray > 200) / (height * width)
        
        # 检测是否包含大量文字区域（通过边缘密度粗略估计）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # 检测是否包含曲线（实验图表特征）
        # 使用梯度分析检测平滑曲线
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        smooth_curve_ratio = np.sum((gradient_magnitude > 30) & (gradient_magnitude < 100)) / (height * width)
        
        # 检测是否包含网格线（实验图表特征）
        horizontal_lines = np.sum(np.abs(sobely) > 100) / (height * width)
        vertical_lines = np.sum(np.abs(sobelx) > 100) / (height * width)
        grid_ratio = min(horizontal_lines, vertical_lines)
        
        # 判断逻辑
        type_scores = {
            "architecture": 0,
            "experiment_result": 0,
            "dataset": 0,
            "other": 0
        }
        
        # OCR文字判断（优先级最高）
        if ocr_text:
            ocr_lower = ocr_text.lower()
            
            # 检查架构关键词
            arch_count = sum(1 for kw in architecture_keywords if kw.lower() in ocr_lower)
            if arch_count > 0:
                type_scores["architecture"] += arch_count * 3
            
            # 检查实验关键词
            exp_count = sum(1 for kw in experiment_keywords if kw.lower() in ocr_lower)
            if exp_count > 0:
                type_scores["experiment_result"] += exp_count * 3
            
            # 检查数据集关键词
            data_count = sum(1 for kw in dataset_keywords if kw.lower() in ocr_lower)
            if data_count > 0:
                type_scores["dataset"] += data_count * 2
        
        # 视觉信息判断（辅助）
        # 实验图表特征：彩色曲线、网格线
        if color_ratio > 0.1 and grid_ratio > 0.05:
            type_scores["experiment_result"] += 2
        
        # 实验图表特征：平滑曲线
        if smooth_curve_ratio > 0.15 and color_ratio > 0.05:
            type_scores["experiment_result"] += 2
        
        # 架构图特征：白底黑字，边缘密度适中
        if white_ratio > 0.5 and edge_density > 0.05 and edge_density < 0.3:
            type_scores["architecture"] += 1
        
        # 数据集特征：包含多个独立图像区域
        # 通过检测图片中的矩形区域（简化处理）
        if color_ratio > 0.2 and edge_density < 0.1:
            type_scores["dataset"] += 1
        
        # 应用优先级规则
        # experiment_result > architecture > dataset > other
        
        # 确定最终类型
        if type_scores["experiment_result"] > 0:
            final_type = "experiment_result"
        elif type_scores["architecture"] > 0:
            final_type = "architecture"
        elif type_scores["dataset"] > 0:
            final_type = "dataset"
        else:
            final_type = "other"
        
        # 构建结果
        if final_type == "architecture":
            # 尝试识别架构组件
            components = []
            if ocr_text:
                for kw in ["AlexNet", "ResNet", "VGG", "GoogLeNet", "Transformer", 
                          "CNN", "Attention", "Encoder", "Decoder", "Embedding",
                          "Classifier", "Feature Extractor"]:
                    if kw.lower() in ocr_text.lower():
                        components.append(kw)
            
            result = {
                "type": "architecture",
                "description": "展示模型网络结构和数据流关系",
                "important_components": components if components else [],
                "code_relevant_information": "需要实现模型架构中的关键组件和模块连接"
            }
        
        elif final_type == "experiment_result":
            # 尝试识别实验相关组件
            components = []
            if ocr_text:
                for kw in ["AlexNet", "ResNet", "VGG", "GoogLeNet", "Transformer",
                          "Accuracy", "Loss", "Precision", "Recall", "F1", "Epoch"]:
                    if kw.lower() in ocr_text.lower():
                        components.append(kw)
            
            result = {
                "type": "experiment_result",
                "description": "展示实验结果和性能对比数据",
                "important_components": components if components else [],
                "code_relevant_information": "需要实现模型训练和评估流程以复现实验结果"
            }
        
        elif final_type == "dataset":
            result = {
                "type": "dataset",
                "description": "展示数据集样本和类别示例",
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