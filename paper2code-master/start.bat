@echo off
chcp 65001 >nul
title Paper2Code - 论文复现验证系统
echo ============================================================
echo   论文复现验证系统 - 启动脚本
echo   纸上得来终觉浅 · Paper2Code
echo ============================================================
echo.

REM 获取脚本所在目录
set SCRIPT_DIR=%~dp0
set BACKEND_CODE=%SCRIPT_DIR%backend\code
set FRONTEND_DIR=%SCRIPT_DIR%frontend

echo [1/2] 启动 Python Flask API (端口 5000)...
start "Paper2Code-Flask" cmd /c "cd /d %BACKEND_CODE% && echo 启动 Flask API... && python api_server.py"

echo     等待 Flask 启动...
timeout /t 5 /nobreak >nul

echo [2/2] 启动 Express 中间层 (端口 3000)...
start "Paper2Code-Express" cmd /c "cd /d %FRONTEND_DIR% && echo 启动 Express 服务器... && node server.js"

echo.
echo ============================================================
echo   启动完成！
echo.
echo   📡 前端页面:  http://localhost:3000
echo   🔗 Flask API:  http://localhost:5000
echo   📋 API 文档:   http://localhost:5000/api/health
echo.
echo   ⚠️  请勿关闭弹出的命令行窗口！
echo   按任意键退出此窗口（不影响后台服务）
echo ============================================================
pause >nul
