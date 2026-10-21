# omni-asr-lite — 零依赖可插拔多引擎并发语音转文字与豆包 LLM 流式整理框架

`omni-asr-lite` 是一套 **零第三方 pip 依赖 (`dependencies = []`)**、原生适配 **Python 3.15 Free-Threaded (`GIL=False`)** 的高性能三阶段语音转文字与 LLM 智能整理框架：

1. **内置搜狗输入法 14 语种/方言高速 ASR 引擎 (`SogouEngine`)**：
   - 采用零依赖 `RSA-3072 OAEP-SHA256` + `AES-256-CBC` 协议、预握手 TLS 1.3 热连接池 (`prewarm`)、线程局部可复用 `libopus` (`complexity=1` + `memoryview` 零拷贝) 与单次 `sendall()` 合并写出；
   - 默认 16 线程并发，单云长视频识别速度达 **`180x ~ 250x` 实时倍率**，支持普通话、粤语、四川话、东北话、武汉话、天津话、南京话、西安话、河北话、贵州话、济南话、英语等 14 种语言与汉语方言。
2. **内置豆包官方免鉴权 LLM 无锁分桶流式整理流水线 (`StreamingLLMOrganizer`)**：
   - 边解码、边识别、边按 7 段定长分桶（无队头阻塞 Zero Head-of-Line Blocking）异步调用豆包官方文本整理接口（`scene=6`），将 90% 的 LLM 耗时完全隐藏在 ASR 识别过程中。
3. **开放 `ASREngineProtocol` 插件扩展架构 (`register_engine`)**：
   - 任何满足 `ASREngineProtocol` 的自定义 ASR 引擎类（继承 `BaseASREngine`）或普通 Python 函数 `fn(pcm_bytes, *, language, hotwords, context) -> str`，只需一行 `asr.register_engine(...)` 即可接入 `OmniScheduler`；
   - 接入后自动获得 **交错工作窃取并发调度、429 限流指数退避熔断、跨引擎零感知故障转移、相对第一梯队尾声准入控制、`0.30s` 长尾对冲冲刺 (`hedged_wins`) 与 `mode="ensemble"` 同段赛马择优** 能力！

---

## 一、 三阶段零阻塞流水线架构

```text
[Stage 1: 本地 CPU 流式解码与实时 VAD 生产]
  ffmpeg 流式管道 -> memoryview.cast('h') 100ms 窗口 RMS -> 静音谷底实时切分 -> 首段 50ms 内入队
                                      │
                                      ▼ (无锁共享队列 + 交错启动)
[Stage 2: 可插拔多引擎并发工作窃取 + 限流自适应熔断 + 长尾对冲冲刺 (OmniScheduler)]
  ├─ SogouEngine (内置默认主力引擎, 16 线程, 14 种语言/方言, 预热 TLS 池)
  └─ CustomEngine_1 .. N (通过 register_engine 挂载的任意外部/自定义 ASR 引擎插件)
     * 429 限流/异常自适应熔断 (Circuit Breaker) + 跨引擎自动故障转移 (Auto-Failover)
     * 尾声动态准入控制 (Tail Admission Control) + >0.30s 长尾对冲冲刺 (Straggler Hedging)
                                      │
                                      ▼ (定长 7 段分桶无锁并发触发)
[Stage 3: 豆包官方 LLM 12 线程无队头阻塞分桶流式整理 (StreamingLLMOrganizer)]
  任意分桶凑齐即刻异步提交豆包 LLM 规整标点、剔除水词并结构化分段，收尾微块 220 字极速收口！
```

---

## 二、 快速上手

### 1. 一行代码极速转写 + 豆包 LLM 智能整理

```python
from omni_asr_lite import transcribe

res = transcribe(
    "lecture.mp4",
    language="zh",                # 支持 "zh" / "yue" / "sichuan" / "dongbei" / "wuhan" / "en" 等 14 种
    hotwords=["曼昆", "机会成本"],
    organize=True,                # 开启 Stage-3 无锁分桶豆包 LLM 智能整理
)

print("智能整理成稿:\n", res.text)
res.save("lecture.md")            # 导出含遥测表的 Markdown 报告
res.save("lecture.srt")           # 导出精确时间轴 SRT 字幕
```

### 2. 接入自定义 ASR 引擎插件参与多线程调度与容灾 (`register_engine`)

#### 方式 A：通过 `BaseASREngine` 子类接入

```python
from omni_asr_lite import BaseASREngine, OmniASR

class MyCloudASREngine(BaseASREngine):
    name = "my_cloud"
    supported_languages = {"zh", "en"}

    def transcribe_pcm(self, pcm_bytes: bytes, *, language: str = "zh", hotwords=None, context="", timeout=None) -> str:
        # 调用你自己的 ASR 接口，返回识别字符串；若遇到 429 限流抛出异常即可自动触发跨引擎容灾
        ...

with OmniASR() as asr:
    asr.register_engine(MyCloudASREngine(), workers=8, initial_rtf=0.04)
    res = asr.transcribe("input.mp4", mode="omni", organize=True)
    print("各引擎分片贡献:", res.engine_stats)
    print("各引擎遥测指标:", res.engine_metrics)
```

#### 方式 B：直接注册一个普通函数

```python
from omni_asr_lite import OmniASR

def custom_asr_fn(pcm_bytes: bytes, *, language: str = "zh", **kwargs) -> str:
    ...

with OmniASR() as asr:
    asr.register_engine("custom_fn", custom_asr_fn, workers=4)
    res = asr.transcribe("input.wav", mode="ensemble")  # 同段多引擎赛马择优
```

### 3. CLI 命令行使用

```powershell
# 查看内置搜狗引擎支持的 14 种语言与汉语方言
python -m omni_asr_lite --list-languages

# 16 线程极速转写长视频 + 豆包 LLM 智能整理，导出 Markdown 讲义
python -m omni_asr_lite input.mp4 -w 16 -f md -o report.md

# 方言识别（粤语 / 四川话 / 东北话 / 武汉话等）并导出 SRT 字幕
python -m omni_asr_lite cantonese.m4a -l yue -f srt -o out.srt
```
