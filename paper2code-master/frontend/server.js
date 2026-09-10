const express = require('express')
const cors = require('cors')
const multer = require('multer')
const path = require('path')
const fs = require('fs')
const http = require('http')
const app = express()
const port = 3000

// 新后端 Flask API 地址
const FLASK_API = process.env.FLASK_API || 'http://localhost:5000'

app.use(cors())
app.use(express.json())
app.use(express.static('./'))

// 修正 multer/busboy 对中文文件名的 latin1 误解码：
// 浏览器以 UTF-8 发送 filename，busboy 默认按 latin1 解码导致中文乱码，
// 这里把 latin1 字节还原回 UTF-8（纯 ASCII 文件名不受影响）。
function fixFilenameEncoding(name) {
    return Buffer.from(name, 'latin1').toString('utf8')
}

// 文件上传配置
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, './uploads')
    },
    filename: function (req, file, cb) {
        file.originalname = fixFilenameEncoding(file.originalname)
        const name = Date.now() + "-" + file.originalname
        cb(null, name)
    }
})
const upload = multer({ storage: storage })

// ======================== 工具函数 ========================

/**
 * 向 Flask API 发送请求
 */
function flaskRequest(method, apiPath, body = null, formData = null) {
    return new Promise((resolve, reject) => {
        const url = new URL(apiPath, FLASK_API)
        const options = {
            method: method,
            hostname: url.hostname,
            port: url.port,
            path: url.pathname + url.search,
            timeout: 60000  // 增加到60秒，因为流水线可能运行很久
        }

        // FormData 上传
        if (formData) {
            const boundary = '----FormBoundary' + Math.random().toString(36).substring(2)
            options.headers = {
                'Content-Type': 'multipart/form-data; boundary=' + boundary
            }

            const req = http.request(options, (res) => {
                let data = ''
                res.on('data', chunk => data += chunk)
                res.on('end', () => {
                    try { resolve(JSON.parse(data)) }
                    catch (e) { resolve({ code: 500, msg: data }) }
                })
            })

            req.on('error', reject)
            req.on('timeout', () => { req.destroy(); reject(new Error('Request timeout')) })

            let bodyParts = []
            for (const [key, value] of Object.entries(formData)) {
                if (value instanceof Object && value.buffer) {
                    bodyParts.push(Buffer.from(`--${boundary}\r\n`))
                    bodyParts.push(Buffer.from(`Content-Disposition: form-data; name="${key}"; filename="${value.filename}"\r\n`))
                    bodyParts.push(Buffer.from(`Content-Type: ${value.contentType || 'application/pdf'}\r\n\r\n`))
                    bodyParts.push(value.buffer)
                    bodyParts.push(Buffer.from('\r\n'))
                } else {
                    bodyParts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${key}"\r\n\r\n${value}\r\n`))
                }
            }
            bodyParts.push(Buffer.from(`--${boundary}--\r\n`))
            req.write(Buffer.concat(bodyParts))
            req.end()

        } else if (body) {
            options.headers = { 'Content-Type': 'application/json' }
            const req = http.request(options, (res) => {
                let data = ''
                res.on('data', chunk => data += chunk)
                res.on('end', () => {
                    try { resolve(JSON.parse(data)) }
                    catch (e) { resolve({ code: 500, msg: data }) }
                })
            })
            req.on('error', reject)
            req.on('timeout', () => { req.destroy(); reject(new Error('Request timeout')) })
            req.write(JSON.stringify(body))
            req.end()

        } else {
            const req = http.request(options, (res) => {
                let data = ''
                res.on('data', chunk => data += chunk)
                res.on('end', () => {
                    try { resolve(JSON.parse(data)) }
                    catch (e) { resolve({ code: 500, msg: data }) }
                })
            })
            req.on('error', reject)
            req.on('timeout', () => { req.destroy(); reject(new Error('Request timeout')) })
            req.end()
        }
    })
}

// ======================== API 接口 (Step1-8) ========================

// 健康检查
app.get('/api/health', async (req, res) => {
    try {
        const flaskResult = await flaskRequest('GET', '/api/health')
        res.json({
            code: 200,
            msg: 'Express + Flask 服务正常 (Step1-8)',
            flask: flaskResult
        })
    } catch (e) {
        res.json({
            code: 200,
            msg: 'Express 服务正常（Flask 后端未连接）',
            flask: { code: 503, msg: e.message }
        })
    }
})

// 获取论文列表
app.get('/list', async (req, res) => {
    try {
        const result = await flaskRequest('GET', '/list')
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 新增文本内容
app.post('/add', async (req, res) => {
    const { content } = req.body
    if (!content || content.trim() === '') {
        return res.json({ code: 400, msg: '内容不能为空' })
    }
    try {
        const result = await flaskRequest('POST', '/add', { content: content.trim() })
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 文件上传
app.post('/upload', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.json({ code: 400, msg: '未选择文件' })
    }

    try {
        const fileBuffer = fs.readFileSync(req.file.path)
        const formData = {
            file: {
                buffer: fileBuffer,
                filename: req.file.originalname,
                contentType: req.file.mimetype
            },
            content: req.body.content || req.file.originalname
        }

        const result = await flaskRequest('POST', '/upload', null, formData)

        // 上传成功后删除本地临时文件
        try { fs.unlinkSync(req.file.path) } catch (e) { }

        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 删除单条 (支持两种路径格式)
app.delete('/del/:id', async (req, res) => {
    try {
        const result = await flaskRequest('DELETE', `/del/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 批量删除（清空全部）
app.delete('/clear', async (req, res) => {
    try {
        const result = await flaskRequest('DELETE', '/clear')
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 获取单条详情
app.get('/detail/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/detail/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// ★ 核心接口：触发完整流水线 (Step1-8)
app.post('/process/:id', async (req, res) => {
    try {
        const result = await flaskRequest('POST', `/process/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 获取处理状态（轮询用）
app.get('/status/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/status/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 获取处理日志
app.get('/logs/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/logs/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 获取 key_information
app.get('/keyinfo/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/keyinfo/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 获取所有步骤状态
app.get('/steps/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/steps/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 运行下一步
app.post('/next/:id', async (req, res) => {
    try {
        const result = await flaskRequest('POST', `/next/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 运行指定步骤
app.post('/run/:id/:step', async (req, res) => {
    try {
        const force = req.query.force === '1'
        const result = await flaskRequest('POST', `/run/${req.params.id}/${req.params.step}?force=${force ? '1' : '0'}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// 重试上一步
app.post('/retry/:id', async (req, res) => {
    try {
        const result = await flaskRequest('POST', `/retry/${req.params.id}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// ★ API Key 管理
app.get('/settings/apikey', async (req, res) => {
    try {
        const result = await flaskRequest('GET', '/settings/apikey')
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

app.post('/settings/apikey', async (req, res) => {
    try {
        const result = await flaskRequest('POST', '/settings/apikey', req.body)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// ★ ZIP 下载（二进制流转发）
app.get('/download/:id', (req, res) => {
    const apiPath = `/download/${encodeURIComponent(req.params.id)}`
    const url = new URL(apiPath, FLASK_API)
    http.get(url, (flaskRes) => {
        if (flaskRes.statusCode !== 200) {
            let body = ''
            flaskRes.on('data', chunk => body += chunk)
            flaskRes.on('end', () => {
                try { res.status(flaskRes.statusCode).json(JSON.parse(body)) }
                catch { res.status(flaskRes.statusCode).json({ code: flaskRes.statusCode, msg: body }) }
            })
            return
        }
        res.set('Content-Type', 'application/zip')
        if (flaskRes.headers['content-disposition']) {
            res.set('Content-Disposition', flaskRes.headers['content-disposition'])
        }
        flaskRes.pipe(res)
    }).on('error', (e) => {
        res.json({ code: 500, msg: '下载失败: ' + e.message })
    })
})

// ★ 获取执行结果
app.get('/result/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/result/${encodeURIComponent(req.params.id)}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// ★ 获取实验配置
app.get('/config/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/config/${encodeURIComponent(req.params.id)}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// ★ 获取依赖
app.get('/deps/:id', async (req, res) => {
    try {
        const result = await flaskRequest('GET', `/deps/${encodeURIComponent(req.params.id)}`)
        res.json(result)
    } catch (e) {
        res.json({ code: 500, msg: 'Flask 后端不可用: ' + e.message })
    }
})

// ======================== 确保 uploads 目录存在 ========================
if (!fs.existsSync('./uploads')) {
    fs.mkdirSync('./uploads')
}

// ======================== 启动服务 ========================
app.listen(port, () => {
    console.log('='.repeat(60))
    console.log('  🚀 论文复现验证系统 - Express 中间层 (Step1-8)')
    console.log('='.repeat(60))
    console.log(`  📡 前端地址: http://localhost:${port}`)
    console.log(`  🔗 Flask API: ${FLASK_API}`)
    console.log('='.repeat(60))
    console.log('  可用接口:')
    console.log('  GET    /list              - 获取论文列表')
    console.log('  POST   /add               - 新增文本')
    console.log('  POST   /upload            - 上传文件')
    console.log('  DELETE /del/:id           - 删除单条')
    console.log('  DELETE /clear             - 清空全部')
    console.log('  POST   /process/:id       - ★ 触发完整流水线 (Step1-8)')
    console.log('  POST   /next/:id          - 运行下一步')
    console.log('  POST   /run/:id/:step     - 运行指定步骤')
    console.log('  POST   /retry/:id         - 重试上一步')
    console.log('  GET    /status/:id        - 获取处理状态')
    console.log('  GET    /steps/:id         - 获取所有步骤状态')
    console.log('  GET    /logs/:id          - 获取处理日志')
    console.log('  GET    /detail/:id        - 获取详情')
    console.log('  GET    /keyinfo/:id       - 获取 key_information')
    console.log('  GET    /api/health        - 健康检查')
    console.log('='.repeat(60))
})
