# Using Deep Learning for Image-Based Plant Disease Detection

复现论文中 Alexnet、Googlenet 在 PlantVillage 上的图像分类，对比不同训练模式与数据划分下的性能。

- 任务类型: image_classification

## 项目结构

```
├── config
│   └── config.yaml
├── data
│   └── dataset.py
├── evaluation
│   └── evaluate.py
├── models
│   ├── __init__.py
│   ├── alexnet.py
│   └── googlenet.py
├── training
│   └── train.py
├── utils
│   └── metrics.py
├── main.py
├── README.md
└── requirements.txt
```

## 环境准备

```bash
pip install -r requirements.txt
```

## 运行方法

```bash
python main.py
```

实验矩阵（数据版本 / 训练模式 / 运行编号）见 `config/config.yaml` 的 `experiment.runs`。

## 复现步骤

1. 安装依赖：`pip install -r requirements.txt`。
2. 按 `config/config.yaml` 中 `data` / `data_source` 的配置准备数据集（标记为 `status: unresolved` 的数据版本需先补齐来源）。
3. 运行入口 `main.py`，按 config 中的实验矩阵逐个 run 复现。
4. 训练产物（checkpoint、日志）的输出位置见 config 的 `checkpoint` / `logging` 配置。
