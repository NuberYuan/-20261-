"""
临时文件管理 HTTP 接口（零依赖，基于标准库 http.server）

把 temp_file_manager 的 list / summary / detail / keep / discard 包装成 REST 接口，
供 Web 前端直接调用（已支持 CORS，方便浏览器跨域访问）。

启动:
    cd code
    python -m temp_file_api --port 8000
    # 需要允许局域网/远程前端访问时：
    python -m temp_file_api --host 0.0.0.0 --port 8000

接口:
    GET    /health              -> {"status": "ok"}
    GET    /summary             -> 汇总所有临时输出（总量/文件数/每篇占用）
    GET    /papers              -> 列出所有论文输出（摘要）
    GET    /papers/{paper}      -> 单次运行完整文件明细
    POST   /papers/{paper}/keep -> 保留原位（不删除）
    DELETE /papers/{paper}      -> 丢弃（删除输出目录）

说明:
    - 保留语义：keep = 原位保留；discard = 删除 data/output/{paper}/ 整个目录。
    - 前端「是否保存本次生成的文件」：保存 -> POST /papers/{paper}/keep；
      不保存 -> DELETE /papers/{paper}。
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from temp_file_manager import TempFileManager
from logger import logger


class TempFileHandler(BaseHTTPRequestHandler):
    server_version = "Paper2CodeTempFileAPI/1.0"
    manager = TempFileManager()

    # =========================
    # 通用工具
    # =========================

    def _send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_result(self, obj):
        """根据结果自动选择 200 / 404。"""
        status = 404 if isinstance(obj, dict) and obj.get("error") == "not_found" else 200
        self._send_json(status, obj)

    # =========================
    # 路由
    # =========================

    def _parts(self):
        path = urlparse(self.path).path.rstrip("/")
        return [unquote(p) for p in path.split("/") if p]

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parts = self._parts()

        if parts == ["health"]:
            return self._send_json(200, {"status": "ok"})
        if parts == ["summary"]:
            return self._send_json(200, self.manager.summary())
        if parts == ["papers"]:
            return self._send_json(200, {"papers": self.manager.list_papers()})
        if len(parts) == 2 and parts[0] == "papers":
            return self._send_result(self.manager.paper_detail(parts[1]))
        return self._send_json(404, {"error": "not_found", "path": self.path})

    def do_POST(self):
        parts = self._parts()
        # /papers/{paper}/keep
        if len(parts) == 3 and parts[0] == "papers" and parts[2] == "keep":
            return self._send_result(self.manager.keep(parts[1]))
        return self._send_json(404, {"error": "not_found", "path": self.path})

    def do_DELETE(self):
        parts = self._parts()
        if len(parts) == 2 and parts[0] == "papers":
            return self._send_result(self.manager.discard(parts[1]))
        return self._send_json(404, {"error": "not_found", "path": self.path})

    def log_message(self, format, *args):
        logger.info("%s - %s" % (self.address_string(), format % args))


def main():
    parser = argparse.ArgumentParser(description="临时文件管理 HTTP 接口")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TempFileHandler)
    logger.info(f"临时文件管理接口已启动: http://{args.host}:{args.port}")
    logger.info("  端点: /health /summary /papers /papers/{paper} /papers/{paper}/keep")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
