"""
统一 API 服务器 - 论文复现验证系统 (Step1-8)

整合 pipeline_controller 和 temp_file_manager 的功能，提供前端期望的 Flask 风格接口。

端口: 5000

提供 RESTful API：
  - 论文管理（基于文件系统的多论文目录结构）
  - 触发 Step1→Step2→Step3→...→Step8 处理流水线
  - 处理状态轮询
  - 结果查询
"""

import json
import os
import sys
import shutil
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# 将 code/ 目录加入 sys.path
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import DATA_INPUT_DIR, DATA_OUTPUT_DIR, LOG_DIR, get_paper_output_dir, get_paper_paths, save_api_key, API_KEY as CURRENT_API_KEY
from pipeline_controller import (
    PipelineController,
    STEP_DEFINITIONS,
    _OUTPUT_REL,
    public_pipeline_status,
    run_remaining_steps,
)
from temp_file_manager import TempFileManager
from logger import get_logger

# ──────────────────────────────────────────────
# Flask 初始化
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logger = get_logger("api_server")

# ──────────────────────────────────────────────
# 状态存储（内存中，用于轮询）
# ──────────────────────────────────────────────
_controller_cache = {}  # paper_name -> PipelineController
_temp_manager = TempFileManager()
_status_lock = threading.Lock()


def get_controller(paper_name: str) -> PipelineController:
    """获取或创建指定论文的控制器"""
    with _status_lock:
        if paper_name not in _controller_cache:
            _controller_cache[paper_name] = PipelineController(paper_name)
        return _controller_cache[paper_name]


def get_current_paper() -> str:
    """获取当前论文名称（从请求参数或默认）"""
    return request.args.get('paper', 'plant')


# ──────────────────────────────────────────────
# 论文管理接口
# ──────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def api_health():
    """健康检查"""
    papers = _temp_manager.list_papers()
    return jsonify({
        "code": 200,
        "msg": "API 运行正常 (Step1-8)",
        "total_papers": len(papers),
        "papers": papers
    })


@app.route("/list", methods=["GET"])
def list_papers():
    """获取论文列表"""
    try:
        papers = _temp_manager.list_papers()
        # 转换为前端期望的格式
        paper_list = []
        for paper_info in papers:
            paper_name = paper_info.get("paper", "")  # 修复：从 "paper" 获取
            paper_dir = get_paper_output_dir(paper_name)
            
            # 检查各步骤的完成状态
            steps_status = {}
            for step_key, _, _ in STEP_DEFINITIONS:
                output_path = paper_dir / _OUTPUT_REL.get(step_key, step_key)
                steps_status[step_key] = {
                    "done": output_path.exists(),
                    "output": str(output_path)
                }
            
            # 获取关键信息
            key_info = None
            key_info_path = paper_dir / "step3" / "key_information.json"
            if key_info_path.exists():
                try:
                    with open(key_info_path, "r", encoding="utf-8") as f:
                        key_info = json.load(f)
                except Exception:
                    pass
            
            completed_count = sum(1 for s in steps_status.values() if s["done"])
            total_count = len(STEP_DEFINITIONS)
            ctrl_status = get_controller(paper_name).status()
            status = public_pipeline_status(ctrl_status)
            last = ctrl_status.get("last_result") or {}
            failed = status == "failed"

            paper_list.append({
                "id": paper_name,
                "content": paper_info.get("paper", ""),
                "paper_name": paper_name,
                "filepath": str(paper_dir / "source.pdf"),
                "status": status,
                "running": bool(ctrl_status.get("running")),
                "current_step": ctrl_status.get("next_step") or _get_current_step(steps_status),
                "progress": int(completed_count * 100 / total_count) if total_count else 0,
                "error": last.get("error") if failed else None,
                "failed_step": last.get("step") if failed else None,
                "keyInformation": key_info,
                "steps_status": steps_status
            })
        
        return jsonify({"code": 200, "data": paper_list})
    except Exception as e:
        logger.exception("获取论文列表失败")
        return jsonify({"code": 500, "msg": f"获取论文列表失败: {str(e)}"})


def _get_current_step(steps_status: dict) -> str:
    """获取当前正在进行的步骤"""
    for step_key, _, _ in STEP_DEFINITIONS:
        if not steps_status.get(step_key, {}).get("done", False):
            return step_key
    return "completed"


# 步骤输出路径映射
_OUTPUT_REL = {key: rel for key, _, rel in STEP_DEFINITIONS}


@app.route("/add", methods=["POST"])
def add_paper_text():
    """添加纯文本记录（创建新论文目录）"""
    try:
        data = request.get_json(silent=True) or {}
        content = data.get("content", "").strip()
        if not content:
            return jsonify({"code": 400, "msg": "内容不能为空"})
        
        # 使用内容的前50个字符作为论文名
        paper_name = content[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
        paper_dir = get_paper_output_dir(paper_name)
        
        # 创建论文目录
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建基本信息文件
        info_file = paper_dir / "paper_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump({
                "content": content,
                "paper_name": paper_name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "text_only"
            }, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "code": 200,
            "msg": "新增成功",
            "id": paper_name,
            "paper_name": paper_name
        })
    except Exception as e:
        logger.exception("添加论文失败")
        return jsonify({"code": 500, "msg": f"添加失败: {str(e)}"})


@app.route("/upload", methods=["POST"])
def upload_paper():
    """上传 PDF 文件"""
    try:
        if 'file' not in request.files:
            return jsonify({"code": 400, "msg": "未选择文件"})
        
        file = request.files['file']
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return jsonify({"code": 400, "msg": "仅支持 PDF 文件"})
        
        content = request.form.get('content', file.filename)
        
        # 使用文件名作为论文名
        paper_name = Path(file.filename).stem[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
        paper_dir = get_paper_output_dir(paper_name)
        
        # 确保目录存在
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "step1" / "images").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step1" / "tables").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step2").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step3" / "generated").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step4").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step5").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step6").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step7" / "generated_project").mkdir(parents=True, exist_ok=True)
        (paper_dir / "step8").mkdir(parents=True, exist_ok=True)
        
        # 直接保存 PDF 到论文输出目录（不往 input/ 写备份）
        source_pdf = paper_dir / "source.pdf"
        file.save(str(source_pdf))
        
        # 创建基本信息文件
        info_file = paper_dir / "paper_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump({
                "content": content,
                "paper_name": paper_name,
                "original_filename": file.filename,
                "pdf_path": str(source_pdf),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "pdf"
            }, f, ensure_ascii=False, indent=2)
        
        # 注意：无需更新 config.INPUT_PDF。PipelineController 会优先读取
        # 论文目录下的 source.pdf（见 pipeline_controller.__init__），
        # config.INPUT_PDF 只是最后回退；且 `from config import INPUT_PDF`
        # 是值拷贝，直接改 config.INPUT_PDF 也不会影响已导入的控制器。

        return jsonify({
            "code": 200,
            "msg": "上传成功",
            "id": paper_name,
            "paper_name": paper_name,
            "filename": file.filename
        })
    except Exception as e:
        logger.exception("上传失败")
        return jsonify({"code": 500, "msg": f"上传失败: {str(e)}"})


@app.route("/del/<paper_name>", methods=["DELETE"])
@app.route("/papers/<paper_name>", methods=["DELETE"])
def delete_paper(paper_name):
    """删除论文"""
    try:
        # 路径遍历校验
        if not paper_name or paper_name in (".", "..") or "/" in paper_name or "\\" in paper_name:
            return jsonify({"code": 400, "msg": f"无效的论文名称: '{paper_name}'"})

        paper_dir = get_paper_output_dir(paper_name)
        # 确保解析后的路径确实在输出目录下
        try:
            paper_dir.resolve().relative_to(Path(DATA_OUTPUT_DIR).resolve())
        except ValueError:
            return jsonify({"code": 400, "msg": f"无效的论文名称: '{paper_name}'"})

        # 检查是否有pipeline正在运行
        with _status_lock:
            if paper_name in _controller_cache:
                controller = _controller_cache[paper_name]
                if controller._lock.locked():
                    return jsonify({"code": 409, "msg": f"论文 '{paper_name}' 正在处理中，无法删除"})

        if paper_dir.exists():
            # 从缓存中移除
            with _status_lock:
                _controller_cache.pop(paper_name, None)
            shutil.rmtree(str(paper_dir), ignore_errors=True)
            return jsonify({"code": 200, "msg": f"论文 '{paper_name}' 已删除"})
        else:
            return jsonify({"code": 404, "msg": f"论文 '{paper_name}' 不存在"})
    except Exception as e:
        logger.exception(f"删除论文 {paper_name} 失败")
        return jsonify({"code": 500, "msg": f"删除失败: {str(e)}"})


@app.route("/clear", methods=["DELETE"])
def clear_all_papers():
    """清空所有论文"""
    try:
        papers = _temp_manager.list_papers()
        for paper_info in papers:
            paper_name = paper_info.get("paper", "")  # 修复：从 "paper" 获取
            paper_dir = get_paper_output_dir(paper_name)
            if paper_dir.exists():
                shutil.rmtree(str(paper_dir), ignore_errors=True)
        
        # 清理控制器缓存
        with _status_lock:
            _controller_cache.clear()
        
        return jsonify({"code": 200, "msg": "已清空全部论文"})
    except Exception as e:
        logger.exception("清空失败")
        return jsonify({"code": 500, "msg": f"清空失败: {str(e)}"})


# ──────────────────────────────────────────────
# 处理流水线接口 (Step1-8)
# ──────────────────────────────────────────────

@app.route("/process/<paper_name>", methods=["POST"])
def trigger_process(paper_name):
    """触发整个流水线处理（所有未完成的步骤）"""
    try:
        controller = get_controller(paper_name)
        
        # 检查是否已在运行
        status = controller.status()
        if status.get("running"):
            return jsonify({"code": 400, "msg": "该论文正在处理中，请等待完成"})
        
        # 获取当前进度
        next_step = controller.next_step()
        if next_step is None:
            return jsonify({"code": 200, "msg": "所有步骤已完成"})
        
        def run_all_steps():
            results = run_remaining_steps(controller.run_next)
            last = results[-1] if results else {}
            if last.get("status") == "error":
                logger.error(
                    f"[流水线] {paper_name} 在 {last.get('step')} 失败: {last.get('error')}"
                )
            else:
                logger.info(f"[流水线] {paper_name} 结束: {last.get('status')}")

        thread = threading.Thread(target=run_all_steps, daemon=True)
        thread.start()
        
        return jsonify({
            "code": 200,
            "msg": f"处理已启动，从 {next_step} 开始",
            "paper": paper_name
        })
    except Exception as e:
        logger.exception(f"启动处理失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"启动失败: {str(e)}"})


@app.route("/next/<paper_name>", methods=["POST"])
def run_next_step(paper_name):
    """运行下一个未完成的步骤（异步执行）"""
    try:
        controller = get_controller(paper_name)

        # 检查是否已在运行
        if controller._lock.locked():
            return jsonify({
                "code": 400,
                "status": "busy",
                "msg": "已有步骤正在运行"
            }), 400

        # 先获取要执行的步骤名（避免竞态条件：后台线程启动后 next_step 可能已变）
        next_step = controller.next_step()
        if next_step is None:
            return jsonify({
                "code": 200,
                "status": "complete",
                "msg": "所有步骤已完成"
            }), 200

        # 异步执行步骤
        def run_in_background():
            try:
                result = controller.run_next()
                logger.info(f"[异步] 步骤执行完成: {result.get('status')}")
            except Exception as e:
                logger.exception(f"[异步] 步骤执行失败")

        thread = threading.Thread(target=run_in_background, daemon=True)
        thread.start()

        # 立即返回，告诉前端开始执行
        return jsonify({
            "code": 200,
            "status": "running",
            "step": next_step,
            "msg": f"步骤 {next_step} 开始执行，请轮询状态"
        }), 200

    except Exception as e:
        logger.exception(f"启动步骤失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"启动失败: {str(e)}"}), 500


@app.route("/run/<paper_name>/<step>", methods=["POST"])
def run_specific_step(paper_name, step):
    """运行指定步骤"""
    try:
        force = request.args.get('force', '0') == '1'
        controller = get_controller(paper_name)
        result = controller.run_step(step, force=force)
        
        http_status = 200 if result.get("status") == "success" else 500
        return jsonify(result), http_status
    except Exception as e:
        logger.exception(f"运行步骤 {step} 失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"运行失败: {str(e)}"})


@app.route("/retry/<paper_name>", methods=["POST"])
def retry_last_step(paper_name):
    """重试上一步"""
    try:
        controller = get_controller(paper_name)
        result = controller.retry()
        
        http_status = 200 if result.get("status") == "success" else 500
        return jsonify(result), http_status
    except Exception as e:
        logger.exception(f"重试失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"重试失败: {str(e)}"})


@app.route("/status/<paper_name>", methods=["GET"])
def get_status(paper_name):
    """获取处理状态 - 格式与前端期望一致"""
    try:
        controller = get_controller(paper_name)
        status = controller.status()
        
        # 转换为前端期望的格式
        completed_count = len(status.get("completed", []))
        total_count = status.get("total", len(STEP_DEFINITIONS))
        progress = int(completed_count * 100 / total_count) if total_count > 0 else 0
        
        current_status = public_pipeline_status(status)
        last = status.get("last_result") or {}

        return jsonify({
            "code": 200,
            "status": current_status,
            "progress": progress,
            "current_step": status.get("next_step", ""),
            "completed": completed_count,
            "total": total_count,
            "error": last.get("error") if current_status == "failed" else None,
            "failed_step": last.get("step") if current_status == "failed" else None,
            "data": status  # 保留完整信息
        })
    except Exception as e:
        logger.exception(f"获取状态失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取状态失败: {str(e)}"})


@app.route("/steps/<paper_name>", methods=["GET"])
def get_steps(paper_name):
    """获取所有步骤状态"""
    try:
        controller = get_controller(paper_name)
        steps = [
            {
                "step": k,
                "label": label,
                "done": controller.is_done(k),
                "output": str(controller.output_path(k))
            }
            for k, label, _ in STEP_DEFINITIONS
        ]
        return jsonify({"code": 200, "data": steps})
    except Exception as e:
        logger.exception(f"获取步骤失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取步骤失败: {str(e)}"})


@app.route("/detail/<paper_name>", methods=["GET"])
def get_detail(paper_name):
    """获取论文详情"""
    try:
        paper_dir = get_paper_output_dir(paper_name)
        if not paper_dir.exists():
            return jsonify({"code": 404, "msg": f"论文 '{paper_name}' 不存在"})
        
        # 读取论文信息文件
        info_file = paper_dir / "paper_info.json"
        info = {}
        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                pass
        
        return jsonify({
            "code": 200,
            "data": {
                "paper_name": paper_name,
                "info": info,
                "dir": str(paper_dir)
            }
        })
    except Exception as e:
        logger.exception(f"获取详情失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取详情失败: {str(e)}"})


@app.route("/keyinfo/<paper_name>", methods=["GET"])
def get_keyinfo(paper_name):
    """获取关键信息"""
    try:
        paper_dir = get_paper_output_dir(paper_name)
        key_info_path = paper_dir / "step3" / "key_information.json"
        
        if not key_info_path.exists():
            return jsonify({"code": 404, "msg": "关键信息文件不存在"})
        
        with open(key_info_path, "r", encoding="utf-8") as f:
            key_info = json.load(f)
        
        return jsonify({"code": 200, "data": key_info})
    except Exception as e:
        logger.exception(f"获取关键信息失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取关键信息失败: {str(e)}"})


@app.route("/logs/<paper_name>", methods=["GET"])
def get_logs(paper_name):
    """获取处理日志"""
    try:
        log_path = LOG_DIR / "app.log"
        if not log_path.exists():
            return jsonify({"code": 200, "data": []})
        
        # 读取最近的日志
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 只返回与该论文相关的日志
        relevant_logs = [line for line in lines if paper_name in line]
        
        return jsonify({"code": 200, "data": relevant_logs[-100:]})  # 最近100条
    except Exception as e:
        logger.exception(f"获取日志失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取日志失败: {str(e)}"})


# ──────────────────────────────────────────────
# API Key 管理
# ──────────────────────────────────────────────

@app.route("/settings/apikey", methods=["GET"])
def get_api_key():
    """获取当前 API Key（脱敏显示）"""
    import config
    key = config.API_KEY
    # 脱敏：只显示前7位和后4位
    if len(key) > 12:
        masked = key[:7] + "*" * (len(key) - 11) + key[-4:]
    else:
        masked = "***"
    return jsonify({"code": 200, "api_key": masked})


@app.route("/settings/apikey", methods=["POST"])
def set_api_key():
    """设置 API Key"""
    import config
    data = request.get_json(force=True) if request.is_json else request.form
    new_key = (data.get("api_key") or "").strip()
    if not new_key:
        return jsonify({"code": 400, "msg": "API Key 不能为空"})
    save_api_key(new_key)
    config.API_KEY = new_key
    logger.info("API Key 已更新")
    return jsonify({"code": 200, "msg": "API Key 已保存"})


# ──────────────────────────────────────────────
# 输出下载与展示
# ──────────────────────────────────────────────

@app.route("/download/<paper_name>", methods=["GET"])
def download_project(paper_name):
    """打包生成项目为 ZIP 并下载"""
    import io, zipfile
    try:
        project_dir = get_paper_output_dir(paper_name) / "step7" / "generated_project"
        if not project_dir.exists():
            return jsonify({"code": 404, "msg": "项目未生成"})

        # 创建 ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in project_dir.rglob("*"):
                if file_path.is_file() and "__pycache__" not in str(file_path):
                    arcname = file_path.relative_to(project_dir)
                    zf.write(file_path, arcname)
        zip_buffer.seek(0)

        from flask import send_file
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{paper_name}_project.zip'
        )
    except Exception as e:
        logger.exception(f"下载失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"下载失败: {str(e)}"})


@app.route("/result/<paper_name>", methods=["GET"])
def get_execution_result(paper_name):
    """获取执行结果（Step8 报告 + 生成报告）"""
    try:
        result = {}

        # Step7 生成报告
        report_path = get_paper_output_dir(paper_name) / "step7" / "generation_report.yaml"
        if report_path.exists():
            import yaml
            with open(report_path, "r", encoding="utf-8") as f:
                result["generation_report"] = yaml.safe_load(f)

        # Step8 执行报告
        exec_report = get_paper_output_dir(paper_name) / "step8" / "execution_report.yaml"
        if exec_report.exists():
            import yaml
            with open(exec_report, "r", encoding="utf-8") as f:
                result["execution_report"] = yaml.safe_load(f)

        # Step8 执行计划
        exec_plan = get_paper_output_dir(paper_name) / "step8" / "execution_plan.yaml"
        if exec_plan.exists():
            import yaml
            with open(exec_plan, "r", encoding="utf-8") as f:
                result["execution_plan"] = yaml.safe_load(f)

        # 执行日志（最近50行相关日志）
        log_path = LOG_DIR / "app.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            relevant = [l.strip() for l in lines if paper_name in l]
            result["logs"] = relevant[-50:]

        return jsonify({"code": 200, "data": result})
    except Exception as e:
        logger.exception(f"获取结果失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取结果失败: {str(e)}"})


@app.route("/config/<paper_name>", methods=["GET"])
def get_paper_config(paper_name):
    """获取实验配置（config.yaml + execution_spec.yaml + 论文简述）"""
    try:
        result = {}

        # config.yaml
        config_path = get_paper_output_dir(paper_name) / "step7" / "generated_project" / "config" / "config.yaml"
        if config_path.exists():
            result["config_yaml"] = config_path.read_text(encoding="utf-8")

        # execution_spec.yaml（入口命令等）
        spec_path = get_paper_output_dir(paper_name) / "step7" / "execution_spec.yaml"
        if spec_path.exists():
            result["execution_spec"] = spec_path.read_text(encoding="utf-8")

        # 论文关键信息（简述）
        key_info_path = get_paper_output_dir(paper_name) / "step3" / "key_information.json"
        if key_info_path.exists():
            import json
            with open(key_info_path, "r", encoding="utf-8") as f:
                key_info = json.load(f)
            # 只取摘要信息
            result["brief"] = {
                "title": key_info.get("title", ""),
                "research_problem": key_info.get("research_problem", "")[:300],
                "methodology": key_info.get("methodology", {}).get("core_idea", "")[:300],
                "dataset": key_info.get("dataset", {}).get("dataset_name", ""),
            }

        return jsonify({"code": 200, "data": result})
    except Exception as e:
        logger.exception(f"获取配置失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取配置失败: {str(e)}"})


@app.route("/deps/<paper_name>", methods=["GET"])
def get_dependencies(paper_name):
    """获取依赖文件（requirements.txt）"""
    try:
        req_path = get_paper_output_dir(paper_name) / "step7" / "generated_project" / "requirements.txt"
        if not req_path.exists():
            return jsonify({"code": 404, "msg": "requirements.txt 不存在"})
        content = req_path.read_text(encoding="utf-8")
        return jsonify({"code": 200, "data": content})
    except Exception as e:
        logger.exception(f"获取依赖失败: {paper_name}")
        return jsonify({"code": 500, "msg": f"获取依赖失败: {str(e)}"})

if __name__ == "__main__":
    print("=" * 60)
    print("  Paper2Code - 统一 API Server (Step1-8)")
    print("=" * 60)
    print(f"  Port:        5000")
    print(f"  Data Input:  {DATA_INPUT_DIR}")
    print(f"  Data Output: {DATA_OUTPUT_DIR}")
    print(f"  Logs:        {LOG_DIR}")
    print("=" * 60)
    print("  接口说明:")
    print("  GET  /list              - 获取论文列表")
    print("  POST /add               - 添加文本论文")
    print("  POST /upload            - 上传PDF论文")
    print("  DELETE /del/<paper>     - 删除论文")
    print("  DELETE /clear           - 清空全部")
    print("  POST /process/<paper>   - 触发完整流水线")
    print("  POST /next/<paper>      - 运行下一步")
    print("  POST /run/<paper>/<step>- 运行指定步骤")
    print("  POST /retry/<paper>     - 重试上一步")
    print("  GET  /status/<paper>    - 获取处理状态")
    print("  GET  /steps/<paper>     - 获取所有步骤状态")
    print("  GET  /detail/<paper>    - 获取论文详情")
    print("  GET  /keyinfo/<paper>   - 获取关键信息")
    print("  GET  /logs/<paper>      - 获取处理日志")
    print("  GET  /api/health        - 健康检查")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=False)
