"""
json_writer.py

负责保存 Step1 输出的 JSON 文件。

Author: Paper2Code
"""

import json
import os

from logger import logger


class JsonWriter:

    def save(self, paper: dict, output_path: str):
        """
        保存 paper.json

        Args:
            paper: 论文解析结果
            output_path: 输出路径
        """

        # 创建目录
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存 JSON
        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                paper,
                f,
                indent=4,
                ensure_ascii=False
            )

        logger.info(f"Saved JSON -> {output_path}")