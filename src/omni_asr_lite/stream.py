# -*- coding: utf-8 -*-
"""
Omni-ASR-Lite 实时推流会话 (`OmniStreamSession`)：
支持向 `feed(pcm_chunk)` 不断推入 16kHz/16bit/单声道 PCM 字节块，
内部按句优先调度最快健康引擎识别，遇到故障自动切换到已注册的备用引擎，并可选在结束时调用豆包 LLM 整理。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator, List, Optional

from .audio import compute_rms
from .codec import compute_delta
from .models import Segment, StreamEvent, TranscriptResult

if TYPE_CHECKING:
    from .client import OmniASR


class OmniStreamSession:
    """多引擎容灾实时流式语音识别会话（Push Stream）"""

    def __init__(
        self,
        client: "OmniASR",
        *,
        language: str = "zh",
        chunk_sec: float = 1.0,
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
        self.chunk_sec = max(0.3, float(chunk_sec))
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
        self._chunk_bytes = max(6400, (raw_chunk // 640) * 640)

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

    def __enter__(self) -> "OmniStreamSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for _ in self.flush():
                pass

    @property
    def _history_text(self) -> str:
        return "".join(s.text for s in self.segments if s.text.strip())

    def _transcribe_with_failover(self, pcm_bytes: bytes) -> tuple[str, str]:
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
        self._utt_pcm_data.extend(pcm_chunk)
        self._utt_rms_sum += rms
        self._utt_rms_cnt += 1

        if rms < self.vad_silence_rms:
            self._utt_silence_sec += chunk_dur
        else:
            self._utt_silence_sec = 0.0

        cur_seq = self._seq
        self._seq += 1
        txt, eng = self._transcribe_with_failover(bytes(self._utt_pcm_data))
        if txt and txt != self._last_partial_text:
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
        t0 = time.perf_counter()
        best, eng = self._transcribe_with_failover(bytes(self._utt_pcm_data))
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
