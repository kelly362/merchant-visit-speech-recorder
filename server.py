#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import pathlib
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sherpa_onnx
import websockets


ROOT = pathlib.Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def log_message(self, fmt, *args):
        logging.info("http %s", fmt % args)


def text_from_result(result):
    return getattr(result, "text", str(result)).strip()


def make_recognizer(args):
    tokens = pathlib.Path(args.tokens)
    encoder = pathlib.Path(args.encoder)
    decoder = pathlib.Path(args.decoder)
    joiner = pathlib.Path(args.joiner)

    missing = [p for p in (tokens, encoder, decoder, joiner) if not p.exists()]
    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing sherpa-onnx model files:\n"
            f"{formatted}\n\n"
            "Run ./scripts/download-model.sh from speech-realtime-demo first."
        )

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tokens),
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=80,
        decoding_method=args.decoding_method,
        provider=args.provider,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
    )


class AsrSession:
    def __init__(self, recognizer, sample_rate):
        self.recognizer = recognizer
        self.sample_rate = sample_rate
        self.stream = recognizer.create_stream()
        self.final_segments = []

    def accept_audio(self, samples):
        self.stream.accept_waveform(self.sample_rate, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        partial = text_from_result(self.recognizer.get_result(self.stream))
        final = None
        if self.recognizer.is_endpoint(self.stream):
            final = partial
            if final:
                self.final_segments.append(final)
            self.recognizer.reset(self.stream)
            partial = ""

        return {
            "type": "result",
            "partial": partial,
            "final": final,
            "transcript": "\n".join(self.final_segments),
        }

    def finish(self):
        tail_padding = np.zeros(int(self.sample_rate * 0.3), dtype=np.float32)
        self.stream.accept_waveform(self.sample_rate, tail_padding)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        tail = text_from_result(self.recognizer.get_result(self.stream))
        if tail:
            self.final_segments.append(tail)

        return {
            "type": "finished",
            "partial": "",
            "final": tail,
            "transcript": "\n".join(self.final_segments),
        }


async def websocket_handler(websocket, recognizer, sample_rate):
    session = AsrSession(recognizer, sample_rate)
    await websocket.send(
        json.dumps(
            {
                "type": "ready",
                "sampleRate": sample_rate,
                "message": "ASR session ready",
            }
        )
    )

    try:
        async for message in websocket:
            if isinstance(message, str):
                payload = json.loads(message)
                if payload.get("type") == "stop":
                    await websocket.send(json.dumps(session.finish(), ensure_ascii=False))
                    break
                continue

            samples = np.frombuffer(message, dtype=np.float32)
            if samples.size == 0:
                continue

            result = session.accept_audio(samples)
            await websocket.send(json.dumps(result, ensure_ascii=False))
    except websockets.ConnectionClosed:
        pass


def start_http_server(port):
    server = ThreadingHTTPServer(("127.0.0.1", port), StaticHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def main():
    parser = argparse.ArgumentParser(description="Local realtime ASR app demo")
    default_model = ROOT / "models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    parser.add_argument("--http-port", type=int, default=5177)
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--decoding-method", default="greedy_search")
    parser.add_argument("--tokens", default=str(default_model / "tokens.txt"))
    parser.add_argument("--encoder", default=str(default_model / "encoder-epoch-99-avg-1.onnx"))
    parser.add_argument("--decoder", default=str(default_model / "decoder-epoch-99-avg-1.onnx"))
    parser.add_argument("--joiner", default=str(default_model / "joiner-epoch-99-avg-1.onnx"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    recognizer = make_recognizer(args)
    start_http_server(args.http_port)

    logging.info("App: http://127.0.0.1:%s", args.http_port)
    logging.info("WebSocket: ws://127.0.0.1:%s/asr", args.ws_port)

    async with websockets.serve(
        lambda ws: websocket_handler(ws, recognizer, args.sample_rate),
        "127.0.0.1",
        args.ws_port,
        max_size=None,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
