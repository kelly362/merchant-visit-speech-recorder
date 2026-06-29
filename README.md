# 商户拜访语音记录

一个用于销售线下拜访商户时记录对话的网页 Demo。

## 在线共享版

把本仓库发布到 GitHub Pages 后，别人直接打开网页即可使用：

- 点击“开始记录”
- 允许浏览器使用麦克风
- 页面实时显示对话文本
- 点击“结束记录”后生成拜访总结

在线共享版使用浏览器内置语音识别能力，推荐使用最新版 Chrome 或 Edge。

## 本地高质量模型版

在本机运行时，页面会使用 `sherpa-onnx` 本地流式 ASR 模型，适合离线测试和模型方案验证。

```bash
cd speech-realtime-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/download-model.sh
python server.py
```

打开：

```text
http://127.0.0.1:5177
```

## 文件说明

- `index.html`: GitHub Pages 入口，会跳转到 `public/`
- `public/index.html`: App 页面
- `public/app.js`: 录音、实时转写、断句标点、总结逻辑
- `public/styles.css`: 页面样式
- `server.py`: 本地 sherpa-onnx WebSocket 服务
- `scripts/download-model.sh`: 下载本地模型

## 注意

- GitHub Pages 共享版不包含大模型文件，不需要安装 Python。
- 本地模型文件和虚拟环境不会提交到仓库。
- 浏览器语音识别通常需要 HTTPS 环境；GitHub Pages 满足这个条件。
