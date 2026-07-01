const TARGET_SAMPLE_RATE = 16000;
const FRAME_BYTES = 1280;
const query = new URLSearchParams(location.search);
const explicitWs = query.get("asr_ws");
const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const defaultWsPort = query.get("ws_port") || "8090";
const WS_URL = explicitWs || `${wsProtocol}//${location.hostname || "127.0.0.1"}:${defaultWsPort}/asr`;
const MAX_SPEAKERS = Math.max(1, Math.min(8, Number(query.get("max_speakers") || 4)));
const SHORT_FRAGMENT_CHARS = Math.max(0, Number(query.get("short_fragment_chars") || 4));
const NEW_SPEAKER_MIN_CHARS = Math.max(SHORT_FRAGMENT_CHARS + 1, Number(query.get("new_speaker_min_chars") || 8));

const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const clearBtn = document.querySelector("#clearBtn");
const copyBtn = document.querySelector("#copyBtn");
const downloadBtn = document.querySelector("#downloadBtn");
const output = document.querySelector("#output");
const statusPill = document.querySelector("#statusPill");
const connectionText = document.querySelector("#connectionText");
const serviceText = document.querySelector("#serviceText");
const timeText = document.querySelector("#timeText");
const meterBars = [...document.querySelectorAll(".meter span")];

let socket;
let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let startedAt = 0;
let timerId;
let sendBuffer = new Int16Array(0);
let transcriptSegments = [];
let partialBySpeaker = new Map();
let lastFinalSpeaker = "";
let rawSpeakerMap = new Map();
let nextSpeakerNumber = 1;

const speakerColors = ["#165dff", "#00b42a", "#ff7d00", "#d91ad9", "#722ed1", "#08979c"];

function setStatus(kind, text) {
  statusPill.className = `status ${kind}`;
  statusPill.querySelector("strong").textContent = text;
}

function setConnection(text) {
  connectionText.textContent = text;
}

function formatDuration(ms) {
  const total = Math.floor(ms / 1000);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function startTimer() {
  startedAt = Date.now();
  timerId = window.setInterval(() => {
    timeText.textContent = formatDuration(Date.now() - startedAt);
  }, 250);
}

function stopTimer() {
  window.clearInterval(timerId);
  timerId = undefined;
}

function speakerIndex(speaker) {
  const match = String(speaker || "说话人1").match(/\d+/);
  return Math.max(0, Number(match ? match[0] : 1) - 1);
}

function canonicalSpeaker(rawSpeaker, text = "", final = false) {
  const key = rawSpeaker || "说话人1";
  if (rawSpeakerMap.has(key)) return rawSpeakerMap.get(key);

  const canCreateSpeaker =
    nextSpeakerNumber <= MAX_SPEAKERS &&
    (rawSpeakerMap.size === 0 || !final || contentLength(text) >= NEW_SPEAKER_MIN_CHARS);

  if (canCreateSpeaker) {
    const speaker = `说话人${nextSpeakerNumber}`;
    rawSpeakerMap.set(key, speaker);
    nextSpeakerNumber += 1;
    return speaker;
  }

  return lastFinalSpeaker || `说话人${Math.max(1, Math.min(nextSpeakerNumber - 1, MAX_SPEAKERS))}`;
}

function colorForSpeaker(speaker) {
  return speakerColors[speakerIndex(speaker) % speakerColors.length];
}

function clearEmpty() {
  const empty = output.querySelector(".empty");
  if (empty) empty.remove();
}

function createLine(speaker, partial = false) {
  clearEmpty();
  const row = document.createElement("article");
  row.className = `line${partial ? " partial-line" : ""}`;

  const tag = document.createElement("span");
  tag.className = "speaker";
  tag.style.background = colorForSpeaker(speaker);
  tag.textContent = speaker || "说话人1";

  const bubble = document.createElement("p");
  bubble.className = `bubble${partial ? " partial" : ""}`;

  row.append(tag, bubble);
  output.appendChild(row);
  output.scrollTop = output.scrollHeight;
  return bubble;
}

function normalizeFinalText(text) {
  return String(text || "").replace(/^[，,。！？；;、\s]+/, "").trim();
}

function contentLength(text) {
  return String(text || "").replace(/[，,。！？；;、\s]/g, "").length;
}

function removePartial(speaker) {
  const bubble = partialBySpeaker.get(speaker);
  if (bubble?.parentElement) bubble.parentElement.remove();
  partialBySpeaker.delete(speaker);
}

function appendFinal(speaker, text) {
  const finalText = normalizeFinalText(text);
  if (!finalText) return;
  let safeSpeaker = canonicalSpeaker(speaker, finalText, true);
  removePartial(safeSpeaker);
  const lastSegment = transcriptSegments[transcriptSegments.length - 1];
  if (lastSegment && lastSegment.speaker !== safeSpeaker && contentLength(finalText) <= SHORT_FRAGMENT_CHARS) {
    safeSpeaker = lastSegment.speaker;
  }
  if (lastSegment && lastSegment.speaker === safeSpeaker) {
    lastSegment.text += finalText;
    lastSegment.bubble.textContent = lastSegment.text;
  } else {
    const bubble = createLine(safeSpeaker, false);
    bubble.textContent = finalText;
    transcriptSegments.push({ speaker: safeSpeaker, text: finalText, bubble });
  }
  lastFinalSpeaker = safeSpeaker;
  output.scrollTop = output.scrollHeight;
}

function replaceTranscript(segments) {
  transcriptSegments = [];
  partialBySpeaker = new Map();
  rawSpeakerMap = new Map();
  nextSpeakerNumber = 1;
  lastFinalSpeaker = "";
  output.innerHTML = "";

  segments.forEach((segment) => {
    const speaker = segment.speaker || "说话人1";
    const text = normalizeFinalText(segment.text);
    if (!text) return;
    const lastSegment = transcriptSegments[transcriptSegments.length - 1];
    if (lastSegment && lastSegment.speaker === speaker) {
      lastSegment.text += text;
      lastSegment.bubble.textContent = lastSegment.text;
    } else {
      const bubble = createLine(speaker, false);
      bubble.textContent = text;
      transcriptSegments.push({ speaker, text, bubble });
    }
    lastFinalSpeaker = speaker;
  });

  if (!transcriptSegments.length) {
    output.innerHTML = '<p class="empty">没有生成可用的最终分段稿。</p>';
  }
  output.scrollTop = output.scrollHeight;
}

function renderPartial(speaker, text) {
  if (!text) return;
  const safeSpeaker = canonicalSpeaker(speaker, text, false);
  let bubble = partialBySpeaker.get(safeSpeaker);
  if (!bubble) {
    bubble = createLine(safeSpeaker, true);
    partialBySpeaker.set(safeSpeaker, bubble);
  }
  bubble.textContent = text;
  output.scrollTop = output.scrollHeight;
}

function handleServerEvent(data) {
  if (data.type === "ready") {
    setStatus(data.phase === "xfyun" ? "live" : "connecting", data.phase === "xfyun" ? "识别中" : "连接中");
    setConnection(data.message || "讯飞服务已连接");
    return;
  }

  if (data.type === "error") {
    setStatus("error", "服务错误");
    setConnection(data.message || data.code || "讯飞服务返回错误");
    return;
  }

  if (data.type === "diarization_status") {
    setStatus("connecting", "分离中");
    setConnection(data.message || "正在生成最终分段稿...");
    return;
  }

  if (data.type === "transcript_replace") {
    replaceTranscript(data.segments || []);
    setStatus("idle", "已修正");
    setConnection(data.message || "已生成最终说话人分离稿。");
    return;
  }

  if (data.type === "finished") {
    setStatus("idle", "已停止");
    if (statusPill.querySelector("strong").textContent !== "已修正") {
      setConnection("本次录音已结束");
    }
    return;
  }

  if (data.type !== "result") return;
  const speaker = data.speaker || "说话人1";
  if (data.is_final) {
    appendFinal(speaker, data.text);
  } else {
    renderPartial(speaker, data.text);
  }
}

function floatTo16kPcm(input, sourceRate) {
  const ratio = sourceRate / TARGET_SAMPLE_RATE;
  const length = Math.floor(input.length / ratio);
  const outputPcm = new Int16Array(length);

  for (let i = 0; i < length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)] || 0));
    outputPcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }

  return outputPcm;
}

function updateMeter(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / Math.max(samples.length, 1));
  const level = Math.min(1, rms * 18);
  meterBars.forEach((bar, index) => {
    bar.style.transform = `scaleY(${Math.max(0.12, level * (0.6 + index * 0.07))})`;
  });
}

function flushFrames(force = false) {
  while (sendBuffer.length * 2 >= FRAME_BYTES) {
    const samplesPerFrame = FRAME_BYTES / 2;
    const frame = sendBuffer.slice(0, samplesPerFrame);
    sendBuffer = sendBuffer.slice(samplesPerFrame);
    if (socket?.readyState === WebSocket.OPEN) socket.send(frame.buffer);
  }

  if (force && sendBuffer.length && socket?.readyState === WebSocket.OPEN) {
    socket.send(sendBuffer.buffer);
    sendBuffer = new Int16Array(0);
  }
}

function queueAudioFrame(floatSamples) {
  const pcm = floatTo16kPcm(floatSamples, audioContext.sampleRate);
  updateMeter(floatSamples);
  const merged = new Int16Array(sendBuffer.length + pcm.length);
  merged.set(sendBuffer);
  merged.set(pcm, sendBuffer.length);
  sendBuffer = merged;
  flushFrames(false);
}

async function startRecording() {
  startBtn.disabled = true;
  clearBtn.disabled = true;
  setStatus("connecting", "连接中");
  setConnection(`连接中转服务：${WS_URL}`);

  socket = new WebSocket(WS_URL);
  socket.binaryType = "arraybuffer";
  const waitForXfyunReady = new Promise((resolve, reject) => {
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleServerEvent(data);
      if (data.type === "ready" && data.phase === "xfyun") resolve();
      if (data.type === "error") reject(new Error(data.message || "讯飞服务连接失败"));
    };
  });
  socket.onerror = () => {
    setStatus("error", "连接失败");
    setConnection("无法连接中转服务，请确认后端已启动。");
  };
  socket.onclose = () => {
    if (!stopBtn.disabled) setConnection("连接已断开");
  };

  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onclose = () => reject(new Error("WebSocket 在启动前关闭"));
  });
  await waitForXfyunReady;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (event) => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    queueAudioFrame(event.inputBuffer.getChannelData(0));
  };
  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);

  stopBtn.disabled = false;
  serviceText.textContent = WS_URL;
  startTimer();
}

async function stopRecording() {
  stopBtn.disabled = true;
  startBtn.disabled = false;
  clearBtn.disabled = false;
  stopTimer();

  try {
    flushFrames(true);
    if (socket?.readyState === WebSocket.OPEN) socket.send("__END__");
  } catch {}

  if (processorNode) {
    processorNode.onaudioprocess = null;
    processorNode.disconnect();
  }
  if (sourceNode) sourceNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (audioContext) await audioContext.close();

  processorNode = undefined;
  sourceNode = undefined;
  mediaStream = undefined;
  audioContext = undefined;
}

function resetTranscript() {
  transcriptSegments = [];
  partialBySpeaker = new Map();
  lastFinalSpeaker = "";
  rawSpeakerMap = new Map();
  nextSpeakerNumber = 1;
  sendBuffer = new Int16Array(0);
  output.innerHTML = '<p class="empty">点击“开始”并允许麦克风权限。多人轮流说话时，结果会按说话人自动分色展示。</p>';
  timeText.textContent = "00:00";
  setStatus("idle", "未连接");
  setConnection("等待开始录音");
}

function transcriptText() {
  return transcriptSegments.map((segment) => `${segment.speaker}：${segment.text}`).join("\n");
}

startBtn.addEventListener("click", () => {
  startRecording().catch(async (error) => {
    console.error(error);
    setStatus("error", "启动失败");
    setConnection(error?.name === "NotAllowedError" ? "麦克风权限被拒绝，请在浏览器中允许麦克风。" : "启动失败，请检查麦克风权限和后端服务。");
    await stopRecording().catch(() => {});
  });
});

stopBtn.addEventListener("click", () => {
  stopRecording().catch(console.error);
});

clearBtn.addEventListener("click", resetTranscript);

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(transcriptText());
  copyBtn.textContent = "已复制";
  window.setTimeout(() => {
    copyBtn.textContent = "复制文字稿";
  }, 1200);
});

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([transcriptText() || "（暂无转写内容）"], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `xfyun-rtasr-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.txt`;
  link.click();
  URL.revokeObjectURL(url);
});

serviceText.textContent = WS_URL;
resetTranscript();
