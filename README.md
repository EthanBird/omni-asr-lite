# omni-asr-lite — 零依赖可插拔多引擎并发语音转文字与豆包 LLM 流式整理框架

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`omni-asr-lite` 是一套 **零第三方 pip 依赖 (`dependencies = []`)**、兼容 Python 3.8 ~ 3.15（含 **Python 3.14t/3.15t Free-Threaded `GIL=False`**）的高性能三阶段语音转文字与 LLM 智能整理框架：

1. **内置搜狗输入法 14 语种/方言高速 ASR 引擎 (`SogouEngine`)**：
   - 采用零依赖 `RSA-3072 OAEP-SHA256` + `AES-256-CBC` 协议（AES 走系统原生 CNG / CommonCrypto / OpenSSL 单次调用，启动时 NIST 向量自检，兜底纯 Python T-Table）；
   - **预升级 WebSocket 热连接池**：后台提前完成「TCP + TLS 1.3 + RSA 密钥协商 + WS Upgrade」，取用时只需加密配置帧即可直接发音频（每段省去 ~90ms Upgrade RTT），5s Idle TTL + Liveness 检测，陈旧连接透明回退重试；
   - **边编码边发送**：线程局部可复用 `libopus` (`complexity=1`) 按 1s 分块编码并即时 `sendall()`，与网络上行 / 服务端解码完全重叠；
   - **多句定稿完整收集**：服务端会对单段音频返回多条 `is_final` 句子，读取至 Close 帧并全部拼接，不再只取第一句而漏字；
   - 支持普通话、粤语、四川话、东北话、武汉话、天津话、南京话、西安话、河北话、贵州话、济南话、英语等 14 种语言与汉语方言；
   - 同时支持 **真增量全双工流式识别 (`open_stream` / `OmniStreamSession`)**：单条 WebSocket 连接上每次 `feed(pcm_chunk)` 仅编码并发送新增的 20ms Opus 帧（**1.0x 音频带宽，零重复重传**），独占编码器保持跨 chunk 连续状态，句段切换前预取下一条热连接，通过非阻塞 `select` 实时接收 `interim_results` 增量文本。
2. **内置豆包官方免鉴权 LLM 无锁分桶流式整理流水线 (`StreamingLLMOrganizer`)**：
   - 边解码、边识别、边按 7 段定长分桶（Zero Head-of-Line Blocking）异步调用豆包文本整理接口（`scene=6`），ASR 进入尾声后自动切换 220 字微块收口；
   - **内置 `1.6s` 投机对冲重试 (Speculative LLM Hedging，>200 字按长度等比放大) + 硬超时自动降级回原文**：彻底消除单条慢连接（如偶发 15~18s 卡顿）拖尾问题；主请求提前失败时立即换连接重试。
3. **开放 `ASREngineProtocol` 插件扩展架构 (`register_engine`)**：
   - 任何满足 `ASREngineProtocol` 的自定义 ASR 引擎类（继承 `BaseASREngine`）或普通 Python 函数 `fn(pcm_bytes, *, language, hotwords, context) -> str`（采用 `inspect.signature` 静态探测签名，绝不误吞用户函数内部 `TypeError`），只需一行 `asr.register_engine(...)` 即可接入 `OmniScheduler`；
   - 接入后自动获得 **交错工作窃取并发调度、429 限流指数退避熔断、跨引擎零感知故障转移、相对第一梯队尾声准入控制、`0.30s` 跨引擎长尾对冲冲刺 (`hedged_wins`) 与 `mode="ensemble"` 同段赛马择优** 能力；
   - **单引擎同样享有长尾治理**：空闲 worker 会对超过「本轮 P90 × 1.5」的慢分片发起 **分裂对冲 (Split Hedging)**——在静音谷底一分为二并发识别（规避服务端对个别整段音频的确定性慢解码），所有分片定稿即返回，不再被落败请求拖住。

---

## 一、 安装与跨平台依赖说明

### 1. 安装 Python 包（零第三方 pip 依赖）

```bash
# 直接通过 GitHub 安装
pip install git+https://github.com/EthanBird/omni-asr-lite.git

# 或克隆仓库后本地安装
git clone https://github.com/EthanBird/omni-asr-lite.git
cd omni-asr-lite
pip install -e .
```

### 2. 系统级动态库与工具要求 (`libopus` & `ffmpeg`)

- **Windows (x64)**：仓库已内置 `src/omni_asr_lite/_bin/opus.dll`（见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) BSD-3 声明），开箱即用；若需解码 `.mp3/.mp4/.m4a/.m4s` 等非 16kHz WAV 格式，请确保 `ffmpeg` 在 `PATH` 中（`winget install Gyan.FFmpeg`）。
- **Linux (Ubuntu / Debian / RHEL)**：请安装系统 `libopus` 与 `ffmpeg`（若缺失 `libopus`，调用时会立即抛出明确的 `RuntimeError` 安装指引，绝不静默返回空字符串）：
  ```bash
  # Ubuntu / Debian
  sudo apt-get update && sudo apt-get install -y libopus0 ffmpeg

  # RHEL / CentOS / Fedora
  sudo dnf install -y opus ffmpeg
  ```
- **macOS**：
  ```bash
  brew install opus ffmpeg
  ```

---

## 二、 实测性能基准（区分「纯 ASR 阶段」与「含 LLM 端到端」）

> **说明**：当开启 `organize=True` 时，长音频的前序分桶会在 ASR 执行期间并发提交给豆包 LLM，但最后的尾桶仍需等待约 `0.5s ~ 2.5s` 的 LLM 生成时间（长音频受豆包服务端吞吐限制）；短音频（`< 30s`）因总分片少于 1 个桶，LLM 耗时与 ASR 耗时为串行相加。若仅需带标点的 ASR 逐字稿，可设置 `organize=False`。

| 测试样本 | 音频时长 (分片数) | **纯 ASR 阶段耗时 (ASR 实时倍率)** | **分桶 LLM 收尾耗时** | **端到端总耗时 (端到端综合倍率)** | 分片统计 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **清华《经济学原理》第 15 集** | `2607.8s` / 43m28s (`239` 段) | **`5.3s` (`~490x` 实时)** | `2.1 ~ 2.5s` | **`7.5 ~ 7.9s` (`~340x` 实时)** | 有声 `239` / 失败 `0` |
| **B 站树洞温泉 Vlog (`BV1brn36iEDN`)** | `576.1s` / 09m36s (`52` 段) | **`1.3 ~ 1.5s` (`~420x` 实时)** | `0.8 ~ 1.1s` | **`2.2 ~ 2.7s` (`~240x` 实时)** | 有声 `50` / 纯BGM片头片尾 `2` / 失败 `0` |
| **2.4s 短句 (`examples/asr_example.wav`)** | `2.4s` (`1` 段) | **`0.12 ~ 0.17s`（热连接）/ `0.32s`（首次冷启动）** | `0.5s` | **`0.7s`** | 有声 `1` / 失败 `0` |

### v1.1 性能优化前后对比（同机同网络、交替 A/B 实测，默认 16 并发）

| 场景 | v1.0 | v1.1 | 主要来源 |
| :--- | :---: | :---: | :--- |
| Vlog 576s 纯 ASR | `1.66 ~ 3.53s`（多数 ~3.4s，被确定性慢分片拖尾） | **`1.28 ~ 1.55s`** | 分裂对冲 + 不再等待落败请求 + 预升级 WS |
| 讲座 2608s 纯 ASR | `6.92 ~ 7.15s` | **`5.31 ~ 5.50s`** | 预升级 WS（单段 p50 `0.42s → 0.32s`）+ 边编码边发送 |
| 讲座 2608s 含 LLM | `8.38 ~ 9.41s` | **`7.5 ~ 7.9s`** | 上述 + LLM 长度自适应对冲（减少约 30% 无谓重复请求） |
| 单段请求 p50 / p90 | `0.43s / 0.58s` | **`0.32s / 0.43s`** | 省去每段 ~90ms Upgrade RTT 与 ~35ms 编码 |
| 讲座逐字稿字数 | `11044` | **`11114+`** | 修复「只取第一条 `is_final`」导致的漏句 |
| AES 配置加密 (1.2KB) | Win `0.21ms` / Linux 纯 Py `8.8ms` | **Win `0.005ms` / Linux OpenSSL 原生** | 单次原生 CBC 调用；纯 Py 兜底亦提速 7x |

> [!TIP]
> 实测在本机网络下将 `engine_workers={"sogou": 24}` / `32`（CLI：`-w 24`）可把讲座纯 ASR 进一步压到 `~3.9s` / `~3.4s` 且未观察到限流；默认仍保持 16 并发以尊重服务方。

---

## 三、 快速上手

### 1. 长音视频转写 + 豆包 LLM 智能整理 (`transcribe`)

```python
from omni_asr_lite import transcribe

res = transcribe(
    "lecture.mp4",
    language="zh",                # 支持 "zh" / "yue" / "sichuan" / "dongbei" / "wuhan" / "en" 等 14 种
    hotwords=["曼昆", "机会成本"],
    organize=True,                # True: 开启豆包 LLM 分桶整理；False: 仅输出搜狗 ASR 原生带标点逐字稿
    strict=False,                 # True: 若存在重试后仍失败的分片则抛出 RuntimeError；False: 发出 UserWarning
)

print("智能整理成稿:\n", res.text)
print("失败分片列表:", res.failed_segments, "| 纯静音/BGM分片列表:", res.silent_segments)
res.save("lecture.md")            # 导出含遥测表的 Markdown 报告
res.save("lecture.srt")           # 导出精确时间轴 SRT 字幕
```

### 2. 真增量全双工实时流式识别 (`transcribe_stream` / `create_stream`)

每个语音句段内仅建立 **1 条** WebSocket 连接，每次 `feed(pcm_chunk)` 仅发送新增的 20ms Opus 帧（1.0x 带宽，零重复重传）：

```python
from omni_asr_lite import transcribe_stream

for ev in transcribe_stream("examples/asr_example.wav", chunk_sec=0.5):
    print(f"[{ev.type.upper():7s}] [{ev.timestamp}] {ev.text} (delta={ev.delta!r})")
```

### 3. 接入自定义 ASR 引擎插件参与多线程调度与容灾 (`register_engine`)

```python
from omni_asr_lite import BaseASREngine, OmniASR

class MyCloudASREngine(BaseASREngine):
    name = "my_cloud"
    supported_languages = {"zh", "en"}

    def transcribe_pcm(self, pcm_bytes: bytes, *, language: str = "zh", hotwords=None, context="", timeout=None) -> str:
        # 调用自定义 ASR 接口返回文本；若抛出 429 限流或网络异常，调度器自动触发熔断并转移至其他健康引擎
        ...

with OmniASR() as asr:
    asr.register_engine(MyCloudASREngine(), workers=8, initial_rtf=0.04)
    res = asr.transcribe("input.mp4", mode="omni", organize=True)
    print("各引擎分片贡献:", res.engine_stats)
    print("各引擎遥测指标:", res.engine_metrics)
```

---

## 四、 免责声明与开源协议

- **开源协议**：本项目代码采用 [MIT License](LICENSE) 开源；内置的 Windows `opus.dll` 遵循 Xiph.Org Foundation 的 [BSD-3-Clause License](THIRD_PARTY_NOTICES.md)。
- **技术研究声明**：本项目内置的默认客户端协议仅用于语音编码、流式并发调度、自适应熔断与对冲算法研究及个人学习测试。请在生产环境中遵守目标服务商的服务条款，或通过 `register_engine()` 挂载您自有的商业授权 ASR 引擎接口。
