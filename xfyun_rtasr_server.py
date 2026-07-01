#!/usr/bin/env python3
import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import pathlib
import re
import ssl
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import numpy as np
import websockets


ROOT = pathlib.Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
XFYUN_HOST = "office-api-ast-dx.iflyaisol.com"
XFYUN_PATH = "/ast/communicate/v1"
SAMPLE_RATE = 16000
FRAME_BYTES = 1280

TAG_RE = re.compile(r"<\|[^|]+?\|>")
TEXT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def log_message(self, fmt, *args):
        logging.info("http %s", fmt % args)


def clean_text(text):
    text = TAG_RE.sub("", str(text or ""))
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[，,。！？；;、\s]+", "", text)
    return text.strip()


def ensure_punctuation(text):
    text = clean_text(text)
    if not text:
        return ""
    if text[-1] in "。！？；.!?;":
        return text.translate(str.maketrans({".": "。", "?": "？", "!": "！", ";": "；"}))
    if text.endswith(("吗", "呢", "么", "多少", "怎样", "如何", "是不是", "能不能", "可不可以")):
        return f"{text}？"
    return f"{text}。"


def normalize_speaker(value):
    if value in (None, ""):
        return "说话人1"
    try:
        index = int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        index = int(match.group(0)) if match else 0
    return f"说话人{max(0, index) + 1}"


def transcript_lines_to_segments(transcript):
    segments = []
    for line in str(transcript or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(说话人\d*)[:：](.+)$", line)
        if match:
            speaker = match.group(1)
            text = match.group(2).strip()
        else:
            speaker = "说话人1"
            text = line
        if text:
            segments.append({"speaker": speaker, "text": text})
    return segments


def pcm16_bytes_to_float32(chunks):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32)
    return (pcm.astype(np.float32) / 32768.0).clip(-1, 1)


def diarize_with_funasr(audio_chunks):
    samples = pcm16_bytes_to_float32(audio_chunks)
    if samples.size < SAMPLE_RATE:
        return []
    import funasr_server

    transcript, _ = funasr_server.transcribe(samples)
    return transcript_lines_to_segments(transcript)


def _json_loads(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _append_segment(segments, speaker, text, is_final):
    text = ensure_punctuation(text) if is_final else clean_text(text)
    if text and TEXT_RE.search(text):
        segments.append(
            {
                "type": "result",
                "speaker": normalize_speaker(speaker),
                "text": text,
                "is_final": bool(is_final),
            }
        )


def _segments_from_cn_st(st):
    is_final = str(st.get("type", "1")) == "0"
    default_speaker = st.get("rl", 0)
    segments = []
    current_speaker = None
    current_words = []

    def flush():
        nonlocal current_words
        if current_words:
            _append_segment(segments, current_speaker if current_speaker is not None else default_speaker, "".join(current_words), is_final)
            current_words = []

    for rt in st.get("rt", []) or []:
        for ws in rt.get("ws", []) or []:
            word_speaker = ws.get("rl")
            for cw in ws.get("cw", []) or []:
                word = cw.get("w", "")
                if not word:
                    continue
                speaker = cw.get("rl", word_speaker if word_speaker is not None else default_speaker)
                if current_speaker is not None and str(speaker) != str(current_speaker):
                    flush()
                current_speaker = speaker
                current_words.append(word)
    flush()
    return segments


def parse_xfyun_message(raw):
    msg = _json_loads(raw)
    if not isinstance(msg, dict):
        return []

    event = msg.get("_event") or msg.get("msg_type") or msg.get("action")
    inner = _json_loads(msg.get("data"))
    inner_action = inner.get("action") if isinstance(inner, dict) else None
    if event in ("started", "handshake_success", "ready") or inner_action == "started":
        return [{"type": "ready", "phase": "xfyun", "message": "讯飞实时语音转写大模型已连接。"}]

    code = str(msg.get("code", "0"))
    if event == "error" or code not in ("0", "None", ""):
        return [
            {
                "type": "error",
                "code": code,
                "message": str(msg.get("desc") or msg.get("message") or "讯飞服务返回错误"),
            }
        ]

    direct_text = msg.get("text") or msg.get("sentence")
    if direct_text:
        return [
            {
                "type": "result",
                "speaker": normalize_speaker(msg.get("speaker") or msg.get("spk") or msg.get("rl")),
                "text": ensure_punctuation(direct_text) if msg.get("is_final", True) else clean_text(direct_text),
                "is_final": bool(msg.get("is_final", True)),
            }
        ]

    data = inner if isinstance(inner, dict) else msg
    st = data.get("cn", {}).get("st") if isinstance(data.get("cn"), dict) else None
    if isinstance(st, dict):
        return _segments_from_cn_st(st)

    results = data.get("result") or data.get("results") or data.get("segments")
    if isinstance(results, dict):
        results = [results]
    if isinstance(results, list):
        parsed = []
        for item in results:
            if isinstance(item, dict):
                _append_segment(
                    parsed,
                    item.get("speaker") or item.get("spk") or item.get("rl"),
                    item.get("text") or item.get("sentence") or "",
                    item.get("is_final", item.get("final", True)),
                )
        return parsed

    return []


def build_xfyun_url(appid, api_key, api_secret):
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S%z")
    params = {
        "accessKeyId": api_key,
        "appId": appid,
        "uuid": uuid.uuid4().hex,
        "utc": now,
        "audio_encode": "pcm_s16le",
        "samplerate": str(SAMPLE_RATE),
        "lang": "autodialect",
        "role_type": "2",
    }
    base = "&".join(f"{key}={quote(str(value), safe='')}" for key, value in sorted(params.items()))
    signature = base64.b64encode(hmac.new(api_secret.encode(), base.encode(), hashlib.sha1).digest()).decode()
    return f"wss://{XFYUN_HOST}{XFYUN_PATH}?{base}&signature={quote(signature, safe='')}"


def get_credentials():
    appid = os.getenv("XFYUN_APPID", "").strip()
    api_key = os.getenv("XFYUN_APIKEY", "").strip()
    api_secret = os.getenv("XFYUN_APISECRET", "").strip()
    missing = [name for name, value in (("XFYUN_APPID", appid), ("XFYUN_APIKEY", api_key), ("XFYUN_APISECRET", api_secret)) if not value]
    if missing:
        raise RuntimeError(f"缺少环境变量：{', '.join(missing)}")
    return appid, api_key, api_secret


async def relay_handler(browser_ws):
    try:
        appid, api_key, api_secret = get_credentials()
    except RuntimeError as exc:
        await browser_ws.send(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
        return

    xfyun_url = build_xfyun_url(appid, api_key, api_secret)
    audio_chunks = []
    await browser_ws.send(
        json.dumps(
            {
                "type": "ready",
                "phase": "relay",
                "sampleRate": SAMPLE_RATE,
                "frameBytes": FRAME_BYTES,
                "message": "浏览器已连接中转服务，正在连接讯飞。",
            },
            ensure_ascii=False,
        )
    )

    try:
        async with websockets.connect(xfyun_url, max_size=None, ping_interval=None) as xfyun_ws:
            await browser_ws.send(json.dumps({"type": "ready", "phase": "xfyun", "message": "讯飞实时语音转写大模型已连接。"}, ensure_ascii=False))

            async def browser_to_xfyun():
                async for message in browser_ws:
                    if isinstance(message, str):
                        if message == "__END__":
                            await xfyun_ws.send(json.dumps({"end": True}, ensure_ascii=False))
                            break
                        payload = _json_loads(message)
                        if isinstance(payload, dict) and payload.get("type") == "stop":
                            await xfyun_ws.send(json.dumps({"end": True}, ensure_ascii=False))
                            break
                        continue
                    audio_chunks.append(bytes(message))
                    await xfyun_ws.send(message)

            async def xfyun_to_browser():
                async for raw in xfyun_ws:
                    for event in parse_xfyun_message(raw):
                        await browser_ws.send(json.dumps(event, ensure_ascii=False))

            done, pending = await asyncio.wait(
                [asyncio.create_task(browser_to_xfyun()), asyncio.create_task(xfyun_to_browser())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except websockets.ConnectionClosed:
        pass
    except Exception as exc:
        logging.exception("relay failed")
        try:
            await browser_ws.send(json.dumps({"type": "error", "message": f"连接讯飞失败：{exc}"}, ensure_ascii=False))
        except websockets.ConnectionClosed:
            pass
    finally:
        try:
            if os.getenv("ENABLE_FINAL_DIARIZATION", "1") != "0" and audio_chunks:
                await browser_ws.send(
                    json.dumps(
                        {"type": "diarization_status", "message": "正在用本地说话人分离模型生成最终分段稿..."},
                        ensure_ascii=False,
                    )
                )
                try:
                    segments = await asyncio.to_thread(diarize_with_funasr, audio_chunks)
                    if segments:
                        await browser_ws.send(
                            json.dumps(
                                {
                                    "type": "transcript_replace",
                                    "source": "funasr",
                                    "message": "已用本地说话人分离模型修正最终分段。",
                                    "segments": segments,
                                },
                                ensure_ascii=False,
                            )
                        )
                except Exception as exc:
                    logging.exception("final diarization failed")
                    await browser_ws.send(
                        json.dumps(
                            {
                                "type": "diarization_status",
                                "message": f"本地说话人分离失败，保留实时转写初稿：{exc}",
                            },
                            ensure_ascii=False,
                        )
                    )
            await browser_ws.send(json.dumps({"type": "finished"}, ensure_ascii=False))
        except websockets.ConnectionClosed:
            pass


def start_http_server(host, port, certfile=None, keyfile=None):
    server = ThreadingHTTPServer((host, port), StaticHandler)
    if certfile and keyfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile, keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def make_ssl_context(certfile, keyfile):
    if not certfile and not keyfile:
        return None
    if not certfile or not keyfile:
        raise RuntimeError("--certfile 和 --keyfile 必须同时提供")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    return context


async def main():
    parser = argparse.ArgumentParser(description="XFYun realtime ASR large-model relay demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=5177)
    parser.add_argument("--ws-port", type=int, default=8090)
    parser.add_argument("--certfile")
    parser.add_argument("--keyfile")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scheme = "https" if args.certfile else "http"
    ws_scheme = "wss" if args.certfile else "ws"
    start_http_server(args.host, args.http_port, args.certfile, args.keyfile)
    logging.info("App: %s://%s:%s", scheme, args.host, args.http_port)
    logging.info("WebSocket: %s://%s:%s/asr", ws_scheme, args.host, args.ws_port)

    async with websockets.serve(
        relay_handler,
        args.host,
        args.ws_port,
        max_size=None,
        ssl=make_ssl_context(args.certfile, args.keyfile),
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
