# -*- coding: utf-8 -*-
"""
Omni-ASR-Lite 真增量全双工流式会话 (`OmniStreamSession`)：
1. 当主引擎支持原生流式连接 (`open_stream()`，如内置 `SogouEngine`) 时：
   - 在每个语音句段内仅建立 **1 条** WebSocket 连接；
   - 每次调用 `feed(pcm_chunk)` 仅编码并发送新增的音频字节（**1.0x 带宽，零重复重传**）；
   - 非阻塞读取服务端实时返回的 `interim_results` 增量事件与 `is_final` 定稿事件。
2. 当使用仅实现 `transcribe_pcm()` 的自定义插件引擎并触发跨引擎容灾时，自动平滑回退至分段识别。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

from .audio import compute_rms
from .codec import compute_delta
from .models import Segment, StreamEvent, TranscriptResult

if TYPE_CHECKING:
    from .client import OmniASR


class OmniStreamSession:
    """真增量全双工流式语音识别会话（Push Stream，1.0x 音频带宽）"""

    def __init__(
        self,
        client: "OmniASR",
        *,
        language: str = "zh",
        chunk_sec: float = 0.5,
        min_utterance_sec: float = 5.0,
        max_utterance_sec: float = 14.0,
        vad_silence_rms: float = 200.0,
        vad_silence_sec: float = 0.5,
        hotwords: Optional[List[str]] = None,
        context: str = "",
        organize: bool = True,
        source_name: str = "<stream>",
    ):
        self.client = client
        self.language = language
        self.chunk_sec = max(0.2, float(chunk_sec))
        self.min_utterance_sec = max(1.5, float(min_utterance_sec))
        self.max_utterance_sec = max(self.min_utterance_sec + 1.0, float(max_utterance_sec))
        self.vad_silence_rms = float(vad_silence_rms)
        self.vad_silence_sec = float(vad_silence_sec)
        self.hotwords = hotwords
        self.context = context
        self.organize = organize
        self.source_name = source_name

        self._active_engines = self.client.scheduler.select_engines_for_language(language)
        raw_chunk = int(round(self.chunk_sec * 16000)) * 2
        self._chunk_bytes = max(3200, (raw_chunk // 640) * 640)

        self._pcm_buf = bytearray()
        self._total_pcm_bytes = 0
        self._t0 = time.perf_counter()

        self.segments: List[Segment] = []
        self._seg_index = 1
        self._seq = 1
        self._utt_start_sec = 0.0
        self._utt_pcm_data = bytearray()
        self._utt_silence_sec = 0.0
        self._utt_rms_sum = 0.0
        self._utt_rms_cnt = 0
        self._last_partial_text = ""

        # 真增量流式连接状态
        self._live_stream: Optional[Any] = None
        self._live_engine_name: str = "sogou"
        self._utt_t0: float = 0.0

    def __enter__(self) -> "OmniStreamSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for _ in self.flush():
                pass
        elif self._live_stream is not None:
            try:
                self._live_stream.close()
            except Exception:
                pass
            self._live_stream = None

    @property
    def _history_text(self) -> str:
        return "".join(s.text for s in self.segments if s.text.strip())

    def _ensure_live_stream(self) -> Optional[Any]:
        if self._live_stream is not None and not getattr(self._live_stream, "closed", False):
            return self._live_stream
        for eng_name in self._active_engines:
            eng = self.client.scheduler.engines.get(eng_name)
            if eng is not None and hasattr(eng, "open_stream"):
                try:
                    self._live_stream = eng.open_stream(
                        language=self.language,
                        hotwords=self.hotwords,
                    )
                    self._live_engine_name = eng_name
                    if not self._utt_t0:
                        self._utt_t0 = time.perf_counter()
                    return self._live_stream
                except Exception:
                    self._live_stream = None
        return None

    def _transcribe_fallback(self, pcm_bytes: bytes) -> tuple[str, str]:
        tried = set()
        for _ in range(max(1, len(self._active_engines))):
            eng_name = self.client.scheduler._pick_failover_engine(self._active_engines, tried)
            tried.add(eng_name)
            try:
                txt = self.client.scheduler.engines[eng_name].transcribe_pcm(
                    pcm_bytes,
                    language=self.language,
                    hotwords=self.hotwords,
                    context=self.context,
                )
                if txt:
                    return txt, eng_name
            except Exception:
                continue
        return "", self._active_engines[0]

    def _step_chunk(self, pcm_chunk: bytes) -> Optional[StreamEvent]:
        rms = compute_rms(pcm_chunk)
        chunk_dur = len(pcm_chunk) / 32000.0
        if not self._utt_t0:
            self._utt_t0 = time.perf_counter()
        self._utt_pcm_data.extend(pcm_chunk)
        self._utt_rms_sum += rms
        self._utt_rms_cnt += 1

        if rms < self.vad_silence_rms:
            self._utt_silence_sec += chunk_dur
        else:
            self._utt_silence_sec = 0.0

        live = self._ensure_live_stream()
        txt = ""
        eng = self._live_engine_name
        if live is not None:
            try:
                evs = live.send_pcm_chunk(pcm_chunk)
                if evs:
                    txt = evs[-1][0]
            except Exception:
                try:
                    live.close()
                except Exception:
                    pass
                self._live_stream = None

        if txt and txt != self._last_partial_text:
            cur_seq = self._seq
            self._seq += 1
            delta = compute_delta(self._last_partial_text, txt)
            self._last_partial_text = txt
            cur_end = self._utt_start_sec + len(self._utt_pcm_data) / 32000.0
            return StreamEvent(
                type="partial",
                segment_index=self._seg_index,
                seq=cur_seq,
                start=round(self._utt_start_sec, 2),
                end=round(cur_end, 2),
                text=txt,
                delta=delta,
                full_text=self._history_text + txt,
                is_final=False,
                engine=eng,
            )
        return None

    def _finalize_utterance(self) -> Optional[StreamEvent]:
        if not self._utt_pcm_data:
            return None
        t0 = self._utt_t0 or time.perf_counter()
        best = ""
        eng = self._live_engine_name

        if self._live_stream is not None:
            try:
                best = (self._live_stream.finish() or "").strip()
            except Exception:
                best = ""
            finally:
                self._live_stream = None

        if not best:
            best, eng = self._transcribe_fallback(bytes(self._utt_pcm_data))
        if not best:
            best = self._last_partial_text

        dt = time.perf_counter() - t0
        cur_end = self._utt_start_sec + len(self._utt_pcm_data) / 32000.0
        avg_rms = self._utt_rms_sum / max(1, self._utt_rms_cnt)
        delta = compute_delta(self._last_partial_text, best)

        seg = Segment(
            index=self._seg_index,
            start=round(self._utt_start_sec, 2),
            end=round(cur_end, 2),
            text=best,
            rms=round(avg_rms, 1),
            engine=eng,
            latency_sec=round(dt, 2),
        )
        self.segments.append(seg)
        ev = StreamEvent(
            type="final",
            segment_index=self._seg_index,
            seq=self._seq,
            start=seg.start,
            end=seg.end,
            text=best,
            delta=delta,
            full_text=self._history_text,
            is_final=True,
            engine=eng,
        )
        self._seg_index += 1
        self._seq = 1
        self._utt_start_sec = cur_end
        self._utt_pcm_data.clear()
        self._utt_silence_sec = 0.0
        self._utt_rms_sum = 0.0
        self._utt_rms_cnt = 0
        self._last_partial_text = ""
        self._utt_t0 = 0.0
        return ev

    def feed(self, pcm_bytes: bytes) -> Iterator[StreamEvent]:
        if not pcm_bytes:
            return
        self._pcm_buf.extend(pcm_bytes)
        self._total_pcm_bytes += len(pcm_bytes)

        while len(self._pcm_buf) >= self._chunk_bytes:
            chunk = bytes(self._pcm_buf[: self._chunk_bytes])
            del self._pcm_buf[: self._chunk_bytes]

            ev = self._step_chunk(chunk)
            if ev is not None:
                yield ev

            utt_dur = len(self._utt_pcm_data) / 32000.0
            if (utt_dur >= self.min_utterance_sec and self._utt_silence_sec >= self.vad_silence_sec) or (
                utt_dur >= self.max_utterance_sec
            ):
                fin_ev = self._finalize_utterance()
                if fin_ev is not None:
                    yield fin_ev

    def flush(self) -> Iterator[StreamEvent]:
        if self._pcm_buf:
            rem = bytes(self._pcm_buf)
            self._pcm_buf.clear()
            ev = self._step_chunk(rem)
            if ev is not None:
                yield ev
        if self._utt_pcm_data:
            fin_ev = self._finalize_utterance()
            if fin_ev is not None:
                yield fin_ev

    @property
    def result(self) -> TranscriptResult:
        t_asr_done = time.perf_counter() - self._t0
        res = TranscriptResult(
            segments=list(self.segments),
            duration=round(self._total_pcm_bytes / 32000.0, 2),
            source=self.source_name,
            mode="stream",
            language=self.language,
            asr_sec=round(t_asr_done, 2),
        )
        if self.organize and res.raw_text:
            t_org0 = time.perf_counter()
            res.organized_text = self.client.organize_text(res.raw_text)
            res.organize_tail_sec = round(time.perf_counter() - t_org0, 2)
        res.elapsed_sec = round(time.perf_counter() - self._t0, 2)
        return res
