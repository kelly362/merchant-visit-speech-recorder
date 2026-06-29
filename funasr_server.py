#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import pathlib
import re
import time

import numpy as np
import soundfile as sf
from funasr import AutoModel
import websockets


ROOT = pathlib.Path(__file__).resolve().parent
TMP_DIR = ROOT / "tmp" / "funasr-server"
SAMPLE_RATE = 16000
TAG_RE = re.compile(r"<\|[^|]+?\|>")

model = None


def get_model():
    global model
    if model is None:
        logging.info("loading FunASR model")
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            spk_model="cam++",
            device="cpu",
            disable_update=True,
        )
    return model


def clean_text(text):
    text = TAG_RE.sub("", text or "")
    return re.sub(r"\s+", "", text).strip()


def ensure_punctuation(text):
    if not text:
        return ""
    if text[-1] in "。！？；":
        return text
    if text.endswith(("吗", "呢", "么", "多少", "怎样", "如何")):
        return f"{text}？"
    return f"{text}。"


def sentence_lines(result):
    if not result:
        return []

    item = result[0]
    sentence_info = item.get("sentence_info") or []
    lines = []

    if len(sentence_info) > 1:
        for sentence in sentence_info:
            text = ensure_punctuation(clean_text(sentence.get("sentence", "")))
            if not text:
                continue
            spk = sentence.get("spk")
            speaker = f"说话人{int(spk) + 1}" if isinstance(spk, int) else "说话人"
            lines.append(f"{speaker}：{text}")

    if not lines:
        text = clean_text(item.get("text", ""))
        chunks = re.findall(r".{1,32}", text)
        lines = [ensure_punctuation(chunk) for chunk in chunks if chunk]

    return lines


def transcribe(samples):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = TMP_DIR / f"visit-{int(time.time() * 1000)}.wav"
    sf.write(audio_path, samples, SAMPLE_RATE)
    result = get_model().generate(input=str(audio_path), batch_size_s=60)
    return "\n".join(sentence_lines(result)), result


class FunAsrSession:
    def __init__(self):
        self.chunks = []
        self.samples_seen = 0
        self.last_notice_at = 0

    def add_audio(self, samples):
        self.chunks.append(samples.copy())
        self.samples_seen += samples.size

    def should_notice(self):
        seconds = self.samples_seen / SAMPLE_RATE
        if seconds - self.last_notice_at >= 3:
            self.last_notice_at = seconds
            return seconds
        return None

    def audio(self):
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks).astype(np.float32)


async def websocket_handler(websocket):
    session = FunAsrSession()
    await websocket.send(
        json.dumps(
            {
                "type": "ready",
                "sampleRate": SAMPLE_RATE,
                "message": "实验语音模型已连接，结束记录后生成分段结果。",
            },
            ensure_ascii=False,
        )
    )

    try:
        async for message in websocket:
            if isinstance(message, str):
                payload = json.loads(message)
                if payload.get("type") == "stop":
                    samples = session.audio()
                    if samples.size == 0:
                        transcript = ""
                    else:
                        transcript, _ = await asyncio.to_thread(transcribe, samples)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "finished",
                                "partial": "",
                                "final": "",
                                "transcript": transcript,
                            },
                            ensure_ascii=False,
                        )
                    )
                    break
                continue

            samples = np.frombuffer(message, dtype=np.float32)
            if samples.size == 0:
                continue
            session.add_audio(samples)
            seconds = session.should_notice()
            if seconds:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "result",
                            "partial": f"实验模型已收到约 {int(seconds)} 秒音频，结束记录后开始转写。",
                            "final": None,
                            "transcript": "",
                        },
                        ensure_ascii=False,
                    )
                )
    except websockets.ConnectionClosed:
        pass


async def main():
    parser = argparse.ArgumentParser(description="FunASR experimental WebSocket backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("FunASR WebSocket: ws://%s:%s/asr", args.host, args.port)

    async with websockets.serve(websocket_handler, args.host, args.port, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
