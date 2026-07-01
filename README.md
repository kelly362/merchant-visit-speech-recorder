# 讯飞实时语音转写大模型测试台

一个用于测试讯飞“实时语音转写大模型”的网页 Demo。浏览器采集麦克风音频，后端中转到讯飞 WebSocket 接口，前端实时展示带标点的多人说话人分离结果。

## 功能

- 实时麦克风录音与转写
- 多说话人展示：`说话人1：...`、`说话人2：...`
- 自动补齐常见中文标点，提升文字稿可读性
- 复制文字稿、下载 `.txt`
- 密钥仅保存在后端环境变量中，不暴露到前端

## 本地启动

```bash
cd speech-realtime-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export XFYUN_APPID="从讯飞控制台复制"
export XFYUN_APIKEY="从讯飞控制台复制"
export XFYUN_APISECRET="从讯飞控制台复制"

python xfyun_rtasr_server.py
```

打开：

```text
http://127.0.0.1:5177
```

本地 `127.0.0.1` 属于浏览器安全上下文，可以直接请求麦克风权限。

## 分享给其他人测试

只上传到 GitHub Pages 不能直接使用完整功能，因为浏览器页面需要连接 Python WebSocket 后端，后端还要安全保存讯飞密钥。要让别人“点开就能用”，需要把本项目部署到一台可访问的 HTTPS 服务器上。

推荐方式：

1. 把代码上传到 GitHub。
2. 在服务器、云主机或 PaaS 上拉取仓库。
3. 在部署平台的环境变量里配置 `XFYUN_APPID`、`XFYUN_APIKEY`、`XFYUN_APISECRET`。
4. 用 HTTPS/WSS 对外暴露页面和 WebSocket 服务。

局域网或服务器部署时，建议使用 HTTPS 页面和 WSS WebSocket，否则大多数浏览器不会允许麦克风权限。

```bash
python xfyun_rtasr_server.py \
  --host 0.0.0.0 \
  --http-port 5177 \
  --ws-port 8090 \
  --certfile /path/to/fullchain.pem \
  --keyfile /path/to/privkey.pem
```

访问：

```text
https://你的域名或IP:5177
```

如果页面和 WebSocket 不在同一个主机或端口，可以用查询参数指定中转地址：

```text
https://你的页面地址/?asr_ws=wss://你的中转地址/asr
```

如果只想先把前端放到 GitHub Pages，也可以让页面连接另一个已经部署好的后端：

```text
https://你的 GitHub Pages 地址/?asr_ws=wss://你的后端域名/asr
```

注意：不要把讯飞密钥写进前端代码或 GitHub 仓库。

## 文件说明

- `public/index.html`：测试站页面
- `public/app.js`：麦克风采集、PCM 分帧、实时结果渲染
- `public/styles.css`：页面样式
- `xfyun_rtasr_server.py`：讯飞实时转写大模型 WebSocket 中转服务
- `tests/test_xfyun_relay.py`：讯飞消息解析、说话人标签、标点测试
- `server.py` / `funasr_server.py`：旧的本地模型实验服务，保留用于对比测试

## 注意

- 讯飞控制台页面会显示密钥，请不要提交到代码或发到群里。
- 多说话人分离依赖讯飞返回的角色信息，测试时建议不同人轮流说完整句子。
- 如果使用自签 HTTPS 证书，测试者需要信任证书；更推荐部署到已有 HTTPS 域名或网关后面。
