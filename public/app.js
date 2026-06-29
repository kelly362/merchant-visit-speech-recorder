const TARGET_SAMPLE_RATE = 16000;
const WS_URL = `ws://${location.hostname || "127.0.0.1"}:8765/asr`;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const USE_LOCAL_ASR = ["127.0.0.1", "localhost", "::1"].includes(location.hostname);

const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const clearBtn = document.querySelector("#clearBtn");
const partialText = document.querySelector("#partialText");
const statusPill = document.querySelector("#statusPill");
const timeText = document.querySelector("#timeText");
const meterBars = [...document.querySelectorAll(".meter span")];
const summaryZone = document.querySelector("#summaryZone");
const summaryText = document.querySelector("#summaryText");
const summaryMeta = document.querySelector("#summaryMeta");
const confirmOverlay = document.querySelector("#confirmOverlay");
const confirmEndBtn = document.querySelector("#confirmEndBtn");
const continueVisitBtn = document.querySelector("#continueVisitBtn");

let audioContext;
let sourceNode;
let processorNode;
let mediaStream;
let socket;
let recognition;
let startedAt = 0;
let timer;
let committedTranscript = "";
let finalizedLines = [];
let pendingEndConfirmation = false;

function setStatus(kind, label) {
  statusPill.className = `status ${kind}`;
  statusPill.querySelector("strong").textContent = label;
}

function formatDuration(ms) {
  const total = Math.floor(ms / 1000);
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function startTimer() {
  startedAt = Date.now();
  timer = window.setInterval(() => {
    timeText.textContent = formatDuration(Date.now() - startedAt);
  }, 250);
}

function stopTimer() {
  window.clearInterval(timer);
  timer = undefined;
}

function downsampleTo16k(input, sourceRate) {
  if (sourceRate === TARGET_SAMPLE_RATE) return input;

  const ratio = sourceRate / TARGET_SAMPLE_RATE;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    let count = 0;

    for (let j = start; j < end; j += 1) {
      sum += input[j];
      count += 1;
    }

    output[i] = count ? sum / count : 0;
  }

  return output;
}

function updateMeter(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sum += samples[i] * samples[i];
  }

  const rms = Math.sqrt(sum / Math.max(samples.length, 1));
  const level = Math.min(1, rms * 18);
  meterBars.forEach((bar, index) => {
    const local = Math.max(0.08, level * (0.55 + index * 0.08));
    bar.style.transform = `scaleY(${local})`;
  });
}

function normalizeSpeechText(text) {
  return text
    .replace(/\s+/g, "")
    .replace(/[,.]/g, "，")
    .replace(/[?？]+/g, "？")
    .replace(/[!！]+/g, "！")
    .replace(/[;；]+/g, "；")
    .replace(/[。]+/g, "。")
    .trim();
}

function punctuateSentence(text, fallbackPunctuation = "。") {
  const cleaned = normalizeSpeechText(text);
  if (!cleaned) return "";
  if (/[。！？；]$/.test(cleaned)) return cleaned;

  if (/^(什么|怎么|如何|为什么|是不是|能不能|可不可以|要不要|有没有|是否|吗|呢)/.test(cleaned)) {
    return `${cleaned}？`;
  }

  if (/(吗|呢|么|多少|怎样|如何)$/.test(cleaned)) {
    return `${cleaned}？`;
  }

  if (/(好的|可以|明白|确认|没问题|行|好)$/.test(cleaned)) {
    return `${cleaned}。`;
  }

  return `${cleaned}${fallbackPunctuation}`;
}

function formatTranscript(rawText) {
  return rawText
    .split(/\n+/)
    .map((line) => punctuateSentence(line))
    .filter(Boolean)
    .join("\n");
}

function renderLiveDialogue(partial = "") {
  const formattedPartial = partial ? punctuateSentence(partial, "，") : "";
  const content = [committedTranscript, formattedPartial].filter(Boolean).join("\n");
  partialText.textContent = content || "等待开始拜访记录...";
  partialText.scrollTop = partialText.scrollHeight;
}

function appendFinalLine(text) {
  const formattedFinal = punctuateSentence(text);
  if (formattedFinal && !finalizedLines.includes(formattedFinal)) {
    finalizedLines.push(formattedFinal);
    committedTranscript = finalizedLines.join("\n");
  }
  renderLiveDialogue();
}

function renderSystemMessage(message) {
  partialText.textContent = message;
  partialText.scrollTop = 0;
}

function getStartupErrorMessage(error) {
  if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
    return "麦克风权限被拒绝。请点击浏览器地址栏左侧的权限图标，允许使用麦克风后刷新页面，再点击“开始记录”。";
  }

  if (error?.name === "NotFoundError" || error?.name === "DevicesNotFoundError") {
    return "没有检测到可用麦克风。请确认电脑已连接麦克风，并在系统设置中允许浏览器访问麦克风。";
  }

  if (error?.name === "NotReadableError" || error?.name === "TrackStartError") {
    return "麦克风暂时不可用。可能被其他应用占用，请关闭其他录音/会议软件后再试。";
  }

  if (error?.message?.includes("启动前已关闭")) {
    return "本地转写连接已关闭。请确认本地服务仍在运行，建议使用 http://127.0.0.1:5177 打开页面。";
  }

  return "无法开始录音，请检查麦克风权限和本地转写服务。";
}

function splitSentences(text) {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[。！？!?；;])|[，,]/)
    .map((item) => item.trim())
    .filter((item) => item.length >= 4);
}

function findSentences(sentences, keywords, limit = 4) {
  const selected = sentences.filter((sentence) =>
    keywords.some((keyword) => sentence.includes(keyword)),
  );
  return selected.slice(0, limit);
}

function unique(items) {
  return [...new Set(items.filter(Boolean))];
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderSummaryCard(title, items, fallback) {
  const list = items.length ? items : [fallback];
  return `
    <article class="summary-card">
      <h3>${escapeHtml(title)}</h3>
      <ul>${list.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </article>
  `;
}

function generateVisitSummary() {
  const text = committedTranscript.trim();
  summaryZone.hidden = false;

  if (!text) {
    summaryMeta.textContent = "暂无有效对话";
    summaryText.innerHTML = renderSummaryCard("总结状态", [], "还没有识别到可总结的拜访内容。");
    return;
  }

  const sentences = splitSentences(text);
  const needs = findSentences(sentences, ["需要", "想", "希望", "能不能", "是否", "怎么", "如何", "问题", "痛点"]);
  const business = findSentences(sentences, ["门店", "商家", "客户", "订单", "销量", "库存", "配送", "核销", "活动", "平台"]);
  const followUps = findSentences(sentences, ["后续", "回访", "联系", "资料", "方案", "报价", "合同", "明天", "下周", "确认", "推进"]);
  const risks = findSentences(sentences, ["担心", "顾虑", "贵", "成本", "不会", "复杂", "慢", "风险", "不确定", "暂时"]);

  const matchedTopics = unique([
    text.includes("价格") || text.includes("报价") ? "价格与报价" : "",
    text.includes("合同") ? "合同与签约" : "",
    text.includes("活动") || text.includes("补贴") ? "活动与补贴" : "",
    text.includes("配送") || text.includes("履约") ? "配送履约" : "",
    text.includes("库存") || text.includes("商品") ? "商品与库存" : "",
    text.includes("培训") || text.includes("操作") ? "操作培训" : "",
  ]);

  summaryMeta.textContent = `${sentences.length || 1} 条对话线索`;
  summaryText.innerHTML = [
    renderSummaryCard("拜访概况", [
      `本次记录时长 ${timeText.textContent}，已形成约 ${text.length} 字原始对话。`,
      matchedTopics.length ? `重点涉及：${matchedTopics.join("、")}。` : "系统已根据当前对话生成初步结构化纪要。",
    ], "已生成本次拜访概况。"),
    renderSummaryCard("商家诉求", needs, "暂未识别到明确诉求，建议继续补充商家的目标、问题和期望。"),
    renderSummaryCard("沟通要点", business, "暂未识别到清晰业务要点，可在后续沟通中确认门店经营、订单、活动或履约信息。"),
    renderSummaryCard("待跟进事项", followUps, "暂未识别到明确待办，建议补充下一次联系时间、资料发送或方案确认事项。"),
    renderSummaryCard("风险与顾虑", risks, "暂未识别到明显风险或顾虑。"),
  ].join("");
}

function showEndVisitConfirm() {
  confirmOverlay.hidden = false;
}

function hideEndVisitConfirm() {
  confirmOverlay.hidden = true;
}

async function startRecording() {
  if (USE_LOCAL_ASR) {
    await startLocalAsrRecording();
  } else {
    startBrowserSpeechRecording();
  }
}

async function startLocalAsrRecording() {
  startBtn.disabled = true;
  clearBtn.disabled = true;
  summaryZone.hidden = true;
  setStatus("connecting", "连接中");
  renderLiveDialogue("正在连接本地转写服务...");

  socket = new WebSocket(WS_URL);
  socket.binaryType = "arraybuffer";

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "ready") {
      setStatus("live", "记录中");
      renderLiveDialogue("开始记录商家沟通...");
      return;
    }

    if (typeof data.transcript === "string") {
      committedTranscript = formatTranscript(data.transcript);
      finalizedLines = committedTranscript.split(/\n+/).filter(Boolean);
    }

    if (data.partial) {
      renderLiveDialogue(data.partial);
    } else if (data.final) {
      appendFinalLine(data.final);
    }

    if (data.type === "finished") {
      setStatus("idle", "未开始");
      renderLiveDialogue();
      if (pendingEndConfirmation) {
        pendingEndConfirmation = false;
        showEndVisitConfirm();
      }
    }
  };

  socket.onerror = () => {
    setStatus("error", "服务异常");
    renderSystemMessage("无法连接本地转写服务，请确认服务已启动。");
    stopRecording(false);
  };

  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onclose = () => reject(new Error("本地转写连接在启动前已关闭"));
  });

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });

  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);

  processorNode.onaudioprocess = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const channel = event.inputBuffer.getChannelData(0);
    const samples = downsampleTo16k(channel, audioContext.sampleRate);
    updateMeter(samples);
    socket.send(samples.buffer);
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);

  stopBtn.disabled = false;
  startTimer();
}

function startBrowserSpeechRecording() {
  if (!SpeechRecognition) {
    setStatus("error", "浏览器不支持");
    renderSystemMessage("当前浏览器不支持网页端实时语音识别。请用最新版 Chrome 或 Edge 打开这个页面。");
    return;
  }

  startBtn.disabled = true;
  stopBtn.disabled = false;
  clearBtn.disabled = true;
  summaryZone.hidden = true;
  setStatus("live", "记录中");
  renderLiveDialogue("正在请求麦克风权限...");

  recognition = new SpeechRecognition();
  recognition.lang = "zh-CN";
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onstart = () => {
    renderLiveDialogue("开始记录商家沟通...");
    startTimer();
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      const text = result[0]?.transcript || "";
      if (result.isFinal) {
        appendFinalLine(text);
      } else {
        interim += text;
      }
    }
    if (interim) renderLiveDialogue(interim);
  };

  recognition.onerror = (event) => {
    setStatus("error", event.error === "not-allowed" ? "麦克风未授权" : "启动失败");
    const message = event.error === "not-allowed"
      ? "麦克风权限被拒绝。请点击浏览器地址栏左侧的权限图标，允许使用麦克风后刷新页面，再点击“开始记录”。"
      : "浏览器语音识别启动失败，请检查麦克风权限，或换用最新版 Chrome/Edge。";
    renderSystemMessage(message);
    stopRecording(false);
  };

  recognition.onend = () => {
    if (!pendingEndConfirmation) return;
    pendingEndConfirmation = false;
    setStatus("idle", "未开始");
    renderLiveDialogue();
    showEndVisitConfirm();
  };

  recognition.start();
}

async function stopRecording(sendStop = true) {
  stopBtn.disabled = true;
  startBtn.disabled = false;
  clearBtn.disabled = false;
  stopTimer();

  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
  }

  if (sourceNode) sourceNode.disconnect();
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  if (audioContext) {
    await audioContext.close();
  }

  if (socket && socket.readyState === WebSocket.OPEN && sendStop) {
    pendingEndConfirmation = true;
    socket.send(JSON.stringify({ type: "stop" }));
  } else if (socket && socket.readyState !== WebSocket.CLOSED) {
    socket.close();
  }

  if (recognition && sendStop) {
    pendingEndConfirmation = true;
    recognition.stop();
  } else if (recognition) {
    recognition.abort();
  }

  processorNode = undefined;
  sourceNode = undefined;
  mediaStream = undefined;
  audioContext = undefined;
  socket = undefined;
  recognition = undefined;
}

startBtn.addEventListener("click", () => {
  startRecording().catch((error) => {
    console.error(error);
    setStatus("error", error?.name === "NotAllowedError" ? "麦克风未授权" : "启动失败");
    renderSystemMessage(getStartupErrorMessage(error));
    stopRecording(false);
  });
});

stopBtn.addEventListener("click", () => {
  stopRecording(true);
});

clearBtn.addEventListener("click", () => {
  committedTranscript = "";
  finalizedLines = [];
  pendingEndConfirmation = false;
  hideEndVisitConfirm();
  summaryZone.hidden = true;
  summaryText.innerHTML = "";
  summaryMeta.textContent = "";
  renderLiveDialogue();
  timeText.textContent = "00:00";
});

confirmEndBtn.addEventListener("click", () => {
  hideEndVisitConfirm();
  setStatus("idle", "已结束");
  generateVisitSummary();
  summaryZone.scrollIntoView({ behavior: "smooth", block: "start" });
});

continueVisitBtn.addEventListener("click", () => {
  hideEndVisitConfirm();
  setStatus("idle", "可继续");
  renderLiveDialogue("可继续补充记录，点击“开始记录”继续本次拜访。");
});

renderLiveDialogue();
