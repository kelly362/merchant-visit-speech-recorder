const TARGET_SAMPLE_RATE = 16000;
const queryParams = new URLSearchParams(location.search);
const REMOTE_ASR_WS_URL = queryParams.get("asr_ws");
const WS_URL = REMOTE_ASR_WS_URL || `ws://${location.hostname || "127.0.0.1"}:8765/asr`;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const USE_STREAM_ASR = Boolean(REMOTE_ASR_WS_URL) || ["127.0.0.1", "localhost", "::1"].includes(location.hostname);
const STREAM_ASR_LABEL = REMOTE_ASR_WS_URL ? "实验语音模型服务" : "本地转写服务";

const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const clearBtn = document.querySelector("#clearBtn");
const customerInput = document.querySelector("#customerInput");
const storeTypeInput = document.querySelector("#storeTypeInput");
const consentInput = document.querySelector("#consentInput");
const visitMeta = document.querySelector("#visitMeta");
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
const copySummaryBtn = document.querySelector("#copySummaryBtn");
const newVisitBtn = document.querySelector("#newVisitBtn");

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
let latestSummaryText = "";

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
  partialText.classList.toggle("is-empty", !content);
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
  partialText.classList.remove("is-empty");
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

function getVisitContext() {
  const customer = customerInput.value.trim() || "未填写客户";
  const storeType = storeTypeInput.value || "未选择类型";
  return { customer, storeType };
}

function updateVisitMeta(state = "尚未开始拜访") {
  const { customer, storeType } = getVisitContext();
  visitMeta.textContent = `${state} · ${customer} · ${storeType}`;
}

function makeTodos(sentences) {
  const followUps = findSentences(sentences, ["后续", "回访", "联系", "资料", "方案", "报价", "合同", "明天", "下周", "确认", "推进"], 5);
  if (followUps.length) {
    return followUps.map((item, index) => ({
      priority: index === 0 ? "高" : "中",
      content: item,
      due: index === 0 ? "24小时内" : "下次拜访前",
    }));
  }

  return [
    { priority: "高", content: "整理本次拜访诉求，补充客户资料和关键异议。", due: "今天内" },
    { priority: "中", content: "向商家同步适配方案、报价或活动政策。", due: "下次沟通前" },
  ];
}

function makeScripts(needs, risks, context) {
  const firstNeed = needs[0] || "您刚才提到的核心需求";
  const firstRisk = risks[0] || "您担心的投入和效果";
  return [
    {
      scenario: "需求确认",
      line: `关于${firstNeed}，我先帮您拆成可落地的步骤，下一次带着方案和您逐项确认。`,
      why: "先复述需求，再给出下一步动作，降低商家的沟通成本。",
    },
    {
      scenario: "异议回应",
      line: `${firstRisk}我理解，我们可以先从${context.storeType}场景里风险最低的一步试起，看数据再扩大。`,
      why: "把顾虑转成小步试点，更容易推进。",
    },
  ];
}

function renderTodoList(todos) {
  return `
    <article class="summary-card wide">
      <h3>待办清单</h3>
      <div class="todo-list">
        ${todos.map((todo) => `
          <div class="todo-item">
            <span class="priority ${escapeHtml(todo.priority)}">${escapeHtml(todo.priority)}</span>
            <div>
              <p>${escapeHtml(todo.content)}</p>
              <small>${escapeHtml(todo.due)}</small>
            </div>
          </div>
        `).join("")}
      </div>
    </article>
  `;
}

function renderScriptList(scripts) {
  return `
    <article class="summary-card wide">
      <h3>下次拜访话术</h3>
      <div class="script-list">
        ${scripts.map((script) => `
          <div class="script-item">
            <small>场景：${escapeHtml(script.scenario)}</small>
            <p>“${escapeHtml(script.line)}”</p>
            <em>${escapeHtml(script.why)}</em>
          </div>
        `).join("")}
      </div>
    </article>
  `;
}

function buildPlainSummary(context, needs, business, commitments, risks, todos, scripts, text) {
  const lines = [
    "【拜访总结】",
    `客户/门店：${context.customer}`,
    `门店类型：${context.storeType}`,
    `记录时长：${timeText.textContent}`,
    "",
    "【客户诉求】",
    ...(needs.length ? needs.map((item) => `· ${item}`) : ["· 暂未识别到明确诉求"]),
    "",
    "【关键信息】",
    ...(business.length ? business.map((item) => `· ${item}`) : ["· 暂未识别到清晰业务要点"]),
    "",
    "【我方承诺】",
    ...(commitments.length ? commitments.map((item) => `· ${item}`) : ["· 暂未识别到明确承诺"]),
    "",
    "【风险点】",
    ...(risks.length ? risks.map((item) => `· ${item}`) : ["· 暂未识别到明显风险或顾虑"]),
    "",
    "【待办】",
    ...todos.map((todo) => `· [${todo.priority}] ${todo.content}（${todo.due}）`),
    "",
    "【下次话术】",
    ...scripts.map((script) => `· ${script.scenario}：${script.line}`),
    "",
    "【完整文字稿】",
    text || "（无）",
  ];
  return lines.join("\n");
}

function generateVisitSummary() {
  const text = committedTranscript.trim();
  const context = getVisitContext();
  summaryZone.hidden = false;

  if (!text) {
    summaryMeta.textContent = "暂无有效对话";
    summaryText.innerHTML = renderSummaryCard("总结状态", [], "还没有识别到可总结的拜访内容。");
    latestSummaryText = "";
    return;
  }

  const sentences = splitSentences(text);
  const needs = findSentences(sentences, ["需要", "想", "希望", "能不能", "是否", "怎么", "如何", "问题", "痛点"]);
  const business = findSentences(sentences, ["门店", "商家", "客户", "订单", "销量", "库存", "配送", "核销", "活动", "平台"]);
  const commitments = findSentences(sentences, ["我帮", "我们会", "可以给", "发给", "提供", "安排", "确认", "推进", "方案", "报价"], 4);
  const risks = findSentences(sentences, ["担心", "顾虑", "贵", "成本", "不会", "复杂", "慢", "风险", "不确定", "暂时"]);
  const todos = makeTodos(sentences);
  const scripts = makeScripts(needs, risks, context);

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
      `客户/门店：${context.customer}，类型：${context.storeType}。`,
      `本次记录时长 ${timeText.textContent}，已形成约 ${text.length} 字原始对话。`,
      matchedTopics.length ? `重点涉及：${matchedTopics.join("、")}。` : "系统已根据当前对话生成初步结构化纪要。",
    ], "已生成本次拜访概况。"),
    renderSummaryCard("客户诉求", needs, "暂未识别到明确诉求，建议继续补充商家的目标、问题和期望。"),
    renderSummaryCard("关键信息", business, "暂未识别到清晰业务要点，可在后续沟通中确认门店经营、订单、活动或履约信息。"),
    renderSummaryCard("我方承诺", commitments, "暂未识别到明确承诺。"),
    renderSummaryCard("风险点", risks, "暂未识别到明显风险或顾虑。"),
    renderTodoList(todos),
    renderScriptList(scripts),
  ].join("");
  latestSummaryText = buildPlainSummary(context, needs, business, commitments, risks, todos, scripts, text);
}

function showEndVisitConfirm() {
  confirmOverlay.hidden = false;
}

function hideEndVisitConfirm() {
  confirmOverlay.hidden = true;
}

async function startRecording() {
  if (!consentInput.checked) {
    setStatus("error", "需授权");
    renderSystemMessage("请先勾选录音授权，并在开始前告知商家本次录音仅用于生成拜访总结。");
    return;
  }

  updateVisitMeta("准备记录");
  if (USE_STREAM_ASR) {
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
  renderLiveDialogue(`正在连接${STREAM_ASR_LABEL}...`);

  socket = new WebSocket(WS_URL);
  socket.binaryType = "arraybuffer";

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "ready") {
      setStatus("live", "记录中");
      updateVisitMeta("记录中");
      renderLiveDialogue(data.message || "开始记录商家沟通...");
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
    renderSystemMessage(`无法连接${STREAM_ASR_LABEL}，请确认服务已启动。`);
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
    updateVisitMeta("记录中");
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
  latestSummaryText = "";
  hideEndVisitConfirm();
  summaryZone.hidden = true;
  summaryText.innerHTML = "";
  summaryMeta.textContent = "";
  renderLiveDialogue();
  updateVisitMeta();
  timeText.textContent = "00:00";
});

confirmEndBtn.addEventListener("click", () => {
  hideEndVisitConfirm();
  setStatus("idle", "已结束");
  updateVisitMeta("已结束");
  generateVisitSummary();
  summaryZone.scrollIntoView({ behavior: "smooth", block: "start" });
});

continueVisitBtn.addEventListener("click", () => {
  hideEndVisitConfirm();
  setStatus("idle", "可继续");
  updateVisitMeta("可继续");
  renderLiveDialogue("可继续补充记录，点击“开始记录”继续本次拜访。");
});

copySummaryBtn.addEventListener("click", async () => {
  if (!latestSummaryText) {
    renderSystemMessage("还没有可复制的总结，请先结束拜访并生成总结。");
    return;
  }

  try {
    await navigator.clipboard.writeText(latestSummaryText);
    const oldText = copySummaryBtn.textContent;
    copySummaryBtn.textContent = "已复制";
    window.setTimeout(() => {
      copySummaryBtn.textContent = oldText;
    }, 1400);
  } catch {
    renderSystemMessage("复制失败，请手动选择总结内容复制。");
  }
});

newVisitBtn.addEventListener("click", () => {
  customerInput.value = "";
  storeTypeInput.value = "品牌";
  consentInput.checked = false;
  clearBtn.click();
  setStatus("idle", "未开始");
});

customerInput.addEventListener("input", () => updateVisitMeta());
storeTypeInput.addEventListener("change", () => updateVisitMeta());

updateVisitMeta();
renderLiveDialogue();
