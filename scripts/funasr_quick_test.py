#!/usr/bin/env python3
import json
import pathlib
import subprocess
import errno

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from funasr import AutoModel


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp" / "funasr-test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELSCOPE_CACHE = pathlib.Path.home() / ".cache" / "modelscope" / "hub" / "models"
SENSEVOICE_CACHE = MODELSCOPE_CACHE / "iic" / "SenseVoiceSmall"
VAD_CACHE = MODELSCOPE_CACHE / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
SPK_CACHE = MODELSCOPE_CACHE / "iic" / "speech_campplus_sv_zh-cn_16k-common"


if not hasattr(errno, "EREMOTEIO"):
    errno.EREMOTEIO = errno.EIO


def cached_model(path, fallback):
    return str(path) if path.exists() else fallback


def synthesize_line(name, text, filename):
    path = OUT_DIR / filename
    if path.exists():
        return path

    subprocess.run(
        ["say", "-v", name, "-o", str(path), text],
        check=True,
    )
    return path


def build_test_audio():
    lines = [
        ("Eddy (中文（中国大陆）)", "您好，我们今天主要看一下门店活动方案和后续合作节奏。", "sales_1.aiff"),
        ("Tingting", "我比较关心费用是多少，以及活动上线以后有没有效果。", "merchant_1.aiff"),
        ("Eddy (中文（中国大陆）)", "这个可以理解，我稍后把报价和案例发给您，我们先从低风险试点开始。", "sales_2.aiff"),
        ("Tingting", "可以，那你明天把资料发我，我们再确认下一步。", "merchant_2.aiff"),
    ]

    segments = []
    target_sr = 16000
    silence = np.zeros(int(target_sr * 0.45), dtype=np.float32)

    for voice, text, filename in lines:
        audio_path = synthesize_line(voice, text, filename)
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            gcd = np.gcd(sr, target_sr)
            data = resample_poly(data, target_sr // gcd, sr // gcd).astype(np.float32)
        segments.extend([data, silence])

    output = OUT_DIR / "merchant_visit_two_speakers.wav"
    sf.write(output, np.concatenate(segments), target_sr)
    return output


def main():
    audio_path = build_test_audio()
    print(f"Test audio: {audio_path}")

    model = AutoModel(
        model=cached_model(SENSEVOICE_CACHE, "iic/SenseVoiceSmall"),
        vad_model=cached_model(VAD_CACHE, "fsmn-vad"),
        spk_model=cached_model(SPK_CACHE, "cam++"),
        device="cpu",
        disable_update=True,
    )
    result = model.generate(input=str(audio_path), batch_size_s=60)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
