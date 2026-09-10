"""
流程控制器 HTTP 接口（零依赖，基于标准库 http.server）

把 pipeline_controller 的 status / next / run 包装成 REST 接口，供 Web 前端调用。
前端每发一次「下一步」指令，控制器就推进一个 step；异常时返回简短摘要，
详细 traceback 由前端读取 logs/app.log。

启动:
    cd code
    python -m pipeline_api --port 8001
    python -m pipeline_api --paper plant --port 8001
    python -m pipeline_api --pdf data/input/论文1.pdf --port 8001   # 指定 PDF
    # 需要允许局域网/远程前端访问时:
    python -m pipeline_api --host 0.0.0.0 --port 8001

接口:
    GET  /health          -> {"status": "ok"}
    GET  /status          -> 当前进度（下一步是什么、哪些已完成）
    GET  /steps           -> 步骤清单（含每步完成状态，供前端渲染）
    POST /next            -> 运行下一个未完成的 step
    POST /retry           -> 重跑上次执行过的 step
    POST /run/{step}      -> 运行指定 step（?force=1 强制重跑）
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_controller import PipelineController, STEP_DEFINITIONS
from logger import logger


class PipelineHandler(BaseHTTPRequestHandler):
    server_version = "Paper2CodePipelineAPI/1.0"
    controller = PipelineController()

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

    def _parts(self):
        path = urlparse(self.path).path.rstrip("/")
        return [unquote(p) for p in path.split("/") if p]

    def _force(self):
        qs = parse_qs(urlparse(self.path).query)
        return qs.get("force", ["0"])[0].lower() in ("1", "true", "yes")

    def _result_status(self, obj):
        """根据运行结果映射 HTTP 状态码。"""
        st = obj.get("status")
        return 409 if st in ("busy", "skipped") else (500 if st == "error" else 200)

    # =========================
    # 路由
    # =========================

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parts = self._parts()
        if parts == ["health"]:
            return self._send_json(200, {"status": "ok"})
        if parts == ["status"]:
            return self._send_json(200, self.controller.status())
        if parts == ["steps"]:
            steps = [
                {
                    "step": k,
                    "label": label,
                    "done": self.controller.is_done(k),
                    "output": str(self.controller.output_path(k)),
                }
                for k, label, _ in STEP_DEFINITIONS
            ]
            return self._send_json(200, {"paper": self.controller.paper, "steps": steps})
        return self._send_json(404, {"error": "not_found", "path": self.path})

    def do_POST(self):
        parts = self._parts()
        if parts == ["next"]:
            result = self.controller.run_next()
            return self._send_json(self._result_status(result), result)
        if parts == ["retry"]:
            result = self.controller.retry()
            return self._send_json(self._result_status(result), result)
        if len(parts) == 2 and parts[0] == "run":
            result = self.controller.run_step(parts[1], force=self._force())
            return self._send_json(self._result_status(result), result)
        return self._send_json(404, {"error": "not_found", "path": self.path})

    def log_message(self, format, *args):
        logger.info("%s - %s" % (self.address_string(), format % args))


def main():
    parser = argparse.ArgumentParser(description="流程控制器 HTTP 接口")
    parser.add_argument("--port", type=int, default=8001, help="监听端口（默认 8001）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--paper", type=str, default=None,
                        help="目标论文名称（输出目录名，默认取自 PDF 文件名）")
    parser.add_argument("--pdf", type=str, default=None,
                        help="输入 PDF 路径（默认自动检测 data/input/ 下第一个 PDF）")
    args = parser.parse_args()

    paper = args.paper or (Path(args.pdf).stem if args.pdf else "plant")
    PipelineHandler.controller = PipelineController(paper=paper, pdf=args.pdf)

    server = ThreadingHTTPServer((args.host, args.port), PipelineHandler)
    logger.info(f"流程控制器接口已启动: http://{args.host}:{args.port} (paper={paper})")
    logger.info("  端点: /health /status /steps /next /retry /run/{step}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
