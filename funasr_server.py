#!/usr/bin/env python3
import argparse
import asyncio
import errno
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
MODELSCOPE_CACHE = pathlib.Path.home() / ".cache" / "modelscope" / "hub" / "models"
SENSEVOICE_CACHE = MODELSCOPE_CACHE / "iic" / "SenseVoiceSmall"
VAD_CACHE = MODELSCOPE_CACHE / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
SPK_CACHE = MODELSCOPE_CACHE / "iic" / "speech_campplus_sv_zh-cn_16k-common"

model = None


if not hasattr(errno, "EREMOTEIO"):
    errno.EREMOTEIO = errno.EIO


def cached_model(path, fallback):
    return str(path) if path.exists() else fallback


def get_model():
    global model
    if model is None:
        logging.info("loading FunASR model")
        model = AutoModel(
            model=cached_model(SENSEVOICE_CACHE, "iic/SenseVoiceSmall"),
            vad_model=cached_model(VAD_CACHE, "fsmn-vad"),
            spk_model=cached_model(SPK_CACHE, "cam++"),
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


def sentence_time_range(sentence):
    timestamps = sentence.get("timestamp")
    if isinstance(timestamps, list) and timestamps:
        starts = []
        ends = []
        for item in timestamps:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                starts.append(float(item[0]))
                ends.append(float(item[1]))
        if starts and ends:
            return min(starts), max(ends)

    start = sentence.get("start")
    end = sentence.get("end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return float(start), float(end)

    return None


def sentence_audio_segment(samples, sentence, sample_rate):
    time_range = sentence_time_range(sentence)
    if time_range is None:
        return None

    start_ms, end_ms = time_range
    start = max(0, int(start_ms * sample_rate / 1000))
    end = min(samples.size, int(end_ms * sample_rate / 1000))
    if end - start < int(sample_rate * 0.18):
        return None
    return samples[start:end]


def speaker_feature(segment, sample_rate):
    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        return None

    segment = segment - float(np.mean(segment))
    energy = float(np.sqrt(np.mean(segment**2)))
    if energy < 1e-4:
        return None

    windowed = segment * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(segment.size, 1 / sample_rate)
    power = spectrum**2
    total_power = float(np.sum(power))
    if total_power <= 0:
        return None

    centroid = float(np.sum(freqs * power) / total_power)
    zero_crossing = float(np.mean(np.abs(np.diff(np.signbit(segment)))))
    return np.array([centroid / 1000, zero_crossing * 10, energy], dtype=np.float32)


def kmeans_two(features, iterations=12):
    if len(features) < 2:
        return [0] * len(features)

    values = np.vstack(features).astype(np.float32)
    normalized = values.copy()
    spread = normalized.std(axis=0)
    spread[spread < 1e-6] = 1
    normalized = (normalized - normalized.mean(axis=0)) / spread

    first = int(np.argmin(normalized[:, 0]))
    second = int(np.argmax(normalized[:, 0]))
    if first == second:
        return [0] * len(features)

    centers = np.vstack([normalized[first], normalized[second]])
    labels = np.zeros(len(features), dtype=np.int64)

    for _ in range(iterations):
        distances = np.linalg.norm(normalized[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for idx in range(2):
            members = normalized[labels == idx]
            if members.size:
                centers[idx] = members.mean(axis=0)

    if len(set(labels.tolist())) < 2:
        return [0] * len(features)

    center_gap = float(np.linalg.norm(centers[0] - centers[1]))
    if center_gap < 0.8:
        return [0] * len(features)

    remap = {}
    ordered = []
    for label in labels.tolist():
        if label not in remap:
            remap[label] = len(remap)
        ordered.append(remap[label])
    return ordered


def infer_speakers_from_audio(sentences, samples, sample_rate):
    if samples is None or len(sentences) < 2:
        return []

    features = []
    valid_positions = []
    for index, sentence in enumerate(sentences):
        segment = sentence_audio_segment(samples, sentence, sample_rate)
        if segment is None:
            continue
        feature = speaker_feature(segment, sample_rate)
        if feature is None:
            continue
        valid_positions.append(index)
        features.append(feature)

    if len(features) < 2:
        return []

    clustered = kmeans_two(features)
    labels = [None] * len(sentences)
    for index, label in zip(valid_positions, clustered):
        labels[index] = label

    last_label = 0
    for index, label in enumerate(labels):
        if label is None:
            labels[index] = last_label
        else:
            last_label = label
    return labels


def sentence_info_from_word_timestamps(item, gap_ms=500):
    words = item.get("words") or []
    timestamps = item.get("timestamp") or []
    if not isinstance(words, list) or not isinstance(timestamps, list):
        return []
    if len(words) < 2 or len(words) != len(timestamps):
        return []

    sentences = []
    current_words = []
    current_timestamps = []
    previous_end = None

    for word, timestamp in zip(words, timestamps):
        if not isinstance(word, str):
            continue
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
            continue

        start = float(timestamp[0])
        end = float(timestamp[1])
        if previous_end is not None and start - previous_end >= gap_ms and current_words:
            sentences.append(
                {
                    "sentence": "".join(current_words),
                    "timestamp": current_timestamps,
                    "start": current_timestamps[0][0],
                    "end": current_timestamps[-1][1],
                }
            )
            current_words = []
            current_timestamps = []

        current_words.append(word)
        current_timestamps.append([start, end])
        previous_end = end

    if current_words:
        sentences.append(
            {
                "sentence": "".join(current_words),
                "timestamp": current_timestamps,
                "start": current_timestamps[0][0],
                "end": current_timestamps[-1][1],
            }
        )

    return sentences if len(sentences) > 1 else []


def sentence_lines(result, samples=None, sample_rate=SAMPLE_RATE):
    if not result:
        return []

    item = result[0]
    sentence_info = item.get("sentence_info") or []
    if len(sentence_info) <= 1:
        sentence_info = sentence_info_from_word_timestamps(item) or sentence_info
    lines = []

    model_speakers = [
        sentence.get("spk")
        for sentence in sentence_info
        if isinstance(sentence.get("spk"), int)
    ]
    has_distinct_model_speakers = len(set(model_speakers)) > 1
    inferred_speakers = []

    if len(sentence_info) > 1 and not has_distinct_model_speakers:
        inferred_speakers = infer_speakers_from_audio(sentence_info, samples, sample_rate)

    if len(sentence_info) > 1:
        for index, sentence in enumerate(sentence_info):
            text = ensure_punctuation(clean_text(sentence.get("sentence", "")))
            if not text:
                continue
            spk = sentence.get("spk")
            if has_distinct_model_speakers and isinstance(spk, int):
                speaker_index = spk
            elif inferred_speakers:
                speaker_index = inferred_speakers[index]
            else:
                speaker_index = None
            speaker = f"说话人{speaker_index + 1}" if isinstance(speaker_index, int) else "说话人"
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
    return "\n".join(sentence_lines(result, samples=samples)), result


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
