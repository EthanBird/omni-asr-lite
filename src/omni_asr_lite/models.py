# -*- coding: utf-8 -*-
"""
omni_asr_lite 数据模型与引擎遥测定义：
- Segment（定稿分段）、StreamEvent（流式事件）、EngineMetrics（单引擎健康度与吞吐遥测）、TranscriptResult（全量结果）
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_ENGINE_WORKERS: Dict[str, int] = {
    "sogou": 16,
}


def fmt_srt_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    s, ms = divmod(total_ms, 1000)
    hrs, rem = divmod(s, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def fmt_vtt_time(seconds: float) -> str:
    return fmt_srt_time(seconds).replace(",", ".")


def fmt_short_time(seconds: float) -> str:
    mins, secs = divmod(max(0.0, seconds), 60)
    return f"{int(mins):02d}:{secs:04.1f}"


@dataclass
class EngineMetrics:
    """单个 ASR 引擎的实时健康度与性能遥测统计"""
    name: str
    workers: int = 0
    succeeded: int = 0
    failed: int = 0
    retried_away: int = 0
    hedged_wins: int = 0
    rate_limited: int = 0
    circuit_trips: int = 0
    total_audio_sec: float = 0.0
    total_latency_sec: float = 0.0
    ewma_rtf: float = 0.04  # 指数移动平均实时率 (耗时/音频时长，越小越快)

    @property
    def avg_speed_x(self) -> float:
        if self.total_latency_sec <= 0.001 or self.total_audio_sec <= 0.001:
            return 0.0
        return round(self.total_audio_sec / self.total_latency_sec, 1)


@dataclass
class Segment:
    """单个语音切分片段的定稿识别结果"""
    index: int
    start: float
    end: float
    text: str
    rms: float = 0.0
    engine: str = "sogou"
    latency_sec: float = 0.0
    failover_chain: List[str] = field(default_factory=list)
    candidates: Dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    @property
    def timestamp(self) -> str:
        return f"{fmt_short_time(self.start)} - {fmt_short_time(self.end)}"


@dataclass
class StreamEvent:
    """流式识别事件对象"""
    type: str
    segment_index: int
    seq: int
    start: float
    end: float
    text: str
    delta: str
    full_text: str
    is_final: bool
    engine: str = "sogou"

    @property
    def timestamp(self) -> str:
        return f"{fmt_short_time(self.start)} - {fmt_short_time(self.end)}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _join_raw_segments(segments: List[Segment]) -> str:
    parts: List[str] = []
    punc_end = tuple("。！？!?；;，,、：:")
    for s in segments:
        t = s.text.strip()
        if not t:
            continue
        if parts and not parts[-1].endswith(punc_end):
            parts[-1] = parts[-1] + "，"
        parts.append(t)
    if parts and not parts[-1].endswith(punc_end):
        parts[-1] = parts[-1] + "。"
    return "".join(parts)


@dataclass
class TranscriptResult:
    """
    完整转写结果对象：
    - `raw_text`: ASR 原始拼接文本
    - `organized_text`: 豆包官方 LLM 无锁分桶流式整理后的书面文本
    - `text`: 若开启了智能整理则返回 `organized_text`，否则返回 `raw_text`
    - `engine_stats`: 各已注册引擎实际完成的分片数量统计
    - `engine_metrics`: 各引擎详细遥测指标（成功率、故障转移次数、限流熔断次数、单引擎实时率等）
    """
    segments: List[Segment] = field(default_factory=list)
    organized_text: str = ""
    duration: float = 0.0
    source: str = ""
    mode: str = "omni"
    language: str = "zh"
    elapsed_sec: float = 0.0
    asr_sec: float = 0.0
    organize_tail_sec: float = 0.0
    engine_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        return _join_raw_segments(self.segments)

    @property
    def text(self) -> str:
        return self.organized_text if self.organized_text.strip() else self.raw_text

    @property
    def lines(self) -> List[str]:
        return [s.text for s in self.segments if s.text.strip()]

    @property
    def engine_stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.segments:
            if s.text.strip():
                counts[s.engine] = counts.get(s.engine, 0) + 1
        return counts

    def to_srt(self) -> str:
        blocks = []
        idx = 1
        for s in self.segments:
            t = s.text.strip()
            if not t:
                continue
            blocks.append(
                f"{idx}\n{fmt_srt_time(s.start)} --> {fmt_srt_time(s.end)}\n{t}\n"
            )
            idx += 1
        return "\n".join(blocks)

    def to_vtt(self) -> str:
        blocks = ["WEBVTT\n"]
        for s in self.segments:
            t = s.text.strip()
            if not t:
                continue
            blocks.append(
                f"{fmt_vtt_time(s.start)} --> {fmt_vtt_time(s.end)}\n{t}\n"
            )
        return "\n".join(blocks)

    def to_markdown(self, title: str = "Omni-ASR-Lite 语音转文字与豆包 LLM 智能整理成稿") -> str:
        stats_str = ", ".join(f"{k}: {v}段" for k, v in self.engine_stats.items()) or "无"
        md = [
            f"# {title}\n",
            f"- **音频来源**: `{self.source or 'PCM Stream'}`",
            f"- **调度模式**: `{self.mode}` (引擎分片分布: `{stats_str}`)",
            f"- **音频时长**: `{self.duration:.1f}s` ({self.duration / 60.0:.1f} 分钟)",
            f"- **端到端总耗时**: `{self.elapsed_sec:.2f}s` (解码+ASR流水线 `{self.asr_sec:.2f}s` + LLM收尾 `{self.organize_tail_sec:.2f}s`)",
            f"- **综合实时倍率**: `{self.duration / max(0.001, self.elapsed_sec):.1f}x` (ASR 阶段 `{self.duration / max(0.001, self.asr_sec):.1f}x`)",
            f"- **有效片段**: `{len(self.lines)} / {len(self.segments)}` 段\n",
        ]
        if self.organized_text.strip():
            md.extend([
                "## 一、 豆包 LLM 智能整理成稿\n",
                self.organized_text.strip(),
                "\n## 二、 ASR 原始逐字稿\n",
                self.raw_text or "*(无识别内容)*",
            ])
        else:
            md.extend([
                "## 一、 完整文字稿\n",
                self.raw_text or "*(无识别内容)*",
            ])

        if self.engine_metrics:
            md.extend([
                "\n## 三、 引擎调度与容灾遥测明细\n",
                "| 引擎名称 | 并发线程 | 成功段数 | 故障转移/重试 | 冲刺胜出 | 限流/熔断次数 | 单引擎速度 |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for name, m in self.engine_metrics.items():
                md.append(
                    f"| `{name}` | `{m.get('workers', 0)}` | `{m.get('succeeded', 0)}` | "
                    f"`{m.get('failed', 0)}` | `{m.get('hedged_wins', 0)}` | "
                    f"`{m.get('rate_limited', 0)}/{m.get('circuit_trips', 0)}` | "
                    f"`{m.get('avg_speed_x', 0.0)}x` |"
                )

        md.extend([
            "\n## 四、 分段时间轴明细\n",
            "| 序号 | 时间轴 | 承接引擎 | 片段耗时 | 识别文本 |",
            "| :---: | :---: | :---: | :---: | :--- |",
        ])
        for s in self.segments:
            txt = s.text.strip() if s.text.strip() else "*(静音/背景音乐)*"
            chain_note = f" (容灾自 `{'->'.join(s.failover_chain[:-1])}`)" if len(s.failover_chain) > 1 else ""
            md.append(f"| {s.index:02d} | `{s.timestamp}` | `{s.engine}`{chain_note} | `{s.latency_sec:.2f}s` | {txt} |")
        return "\n".join(md) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "mode": self.mode,
            "language": self.language,
            "duration": round(self.duration, 2),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "asr_sec": round(self.asr_sec, 2),
            "organize_tail_sec": round(self.organize_tail_sec, 2),
            "engine_stats": self.engine_stats,
            "engine_metrics": self.engine_metrics,
            "text": self.text,
            "raw_text": self.raw_text,
            "organized_text": self.organized_text,
            "segments": [asdict(s) for s in self.segments],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str, fmt: Optional[str] = None) -> str:
        if not fmt:
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            fmt = ext if ext in ("txt", "srt", "vtt", "json", "md") else "txt"
        fmt = fmt.lower()
        if fmt == "srt":
            content = self.to_srt()
        elif fmt == "vtt":
            content = self.to_vtt()
        elif fmt == "json":
            content = self.to_json()
        elif fmt in ("md", "markdown"):
            content = self.to_markdown()
        else:
            content = self.text + "\n"
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
