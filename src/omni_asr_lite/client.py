# -*- coding: utf-8 -*-
"""
Omni-ASR-Lite 顶层客户端 (`OmniASR`)：
- 默认内置搜狗输入法 14 语种/方言高速 ASR 引擎 (`sogou`，默认 16 线程并发 + 预热 TLS 连接池)
- 内置豆包官方免鉴权 LLM 无锁分桶流式整理 (`StreamingLLMOrganizer`)
- 提供开放插件注册接口 `register_engine(...)`：外部满足 `ASREngineProtocol` 的引擎注册后，
  自动加入多线程动态工作窃取、429 限流熔断、跨引擎故障转移、长尾对冲冲刺与 `ensemble` 多引擎择优！
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Iterator, List, Optional, Union

from .audio import (
    compute_rms,
    iter_pcm16k_chunks,
    iter_vad_segments_streaming,
    load_pcm16k,
)
from .engines import ASREngineProtocol, BaseASREngine, SOGOU_LANGUAGES
from .models import Segment, StreamEvent, TranscriptResult
from .scheduler import OmniScheduler
from .stream import OmniStreamSession


class OmniASR:
    """
    零依赖可插拔多引擎并发语音识别 + 豆包官方 LLM 无锁分桶流式整理客户端
    """

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
        self.scheduler = OmniScheduler(timeout=timeout)

    @property
    def languages(self) -> Dict[str, str]:
        return dict(SOGOU_LANGUAGES)

    def register_engine(
        self,
        engine_or_name: Union[str, ASREngineProtocol, BaseASREngine],
        fn: Optional[Callable[..., str]] = None,
        *,
        workers: int = 8,
        initial_rtf: float = 0.05,
        supported_languages: Optional[Iterable[str]] = None,
    ) -> ASREngineProtocol:
        """
        注册一个符合 `ASREngineProtocol` 的自定义 ASR 引擎实例或函数，使其参与多线程抢占、熔断容灾与长尾冲刺。
        """
        return self.scheduler.register_engine(
            engine_or_name,
            fn,
            workers=workers,
            initial_rtf=initial_rtf,
            supported_languages=supported_languages,
        )

    def unregister_engine(self, name: str) -> None:
        self.scheduler.unregister_engine(name)

    def list_engines(self) -> List[str]:
        return self.scheduler.list_engines()

    def close(self) -> None:
        self.scheduler.close()

    def __enter__(self) -> "OmniASR":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def iter_pcm16k_chunks(
        self,
        source: Union[str, os.PathLike, bytes, Iterable[bytes]],
        chunk_bytes: int = 64000,
    ) -> Iterator[bytes]:
        return iter_pcm16k_chunks(source, chunk_bytes=chunk_bytes, ffmpeg_path=self.ffmpeg_path)

    def load_pcm16k(self, source: Union[str, os.PathLike, bytes]) -> bytes:
        return load_pcm16k(source, ffmpeg_path=self.ffmpeg_path)

    def organize_text(
        self,
        text: str,
        *,
        max_chunk_chars: int = 420,
        workers: int = 10,
    ) -> str:
        return self.scheduler.organizer.organize_text(
            text, max_chunk_chars=max_chunk_chars, workers=workers
        )

    # -------------------- 流式接口 --------------------
    def create_stream(
        self,
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
    ) -> OmniStreamSession:
        return OmniStreamSession(
            self,
            language=language,
            chunk_sec=chunk_sec,
            min_utterance_sec=min_utterance_sec,
            max_utterance_sec=max_utterance_sec,
            vad_silence_rms=vad_silence_rms,
            vad_silence_sec=vad_silence_sec,
            hotwords=hotwords,
            context=context,
            organize=organize,
            source_name=source_name,
        )

    def transcribe_stream(
        self,
        source: Union[str, os.PathLike, bytes, Iterable[bytes]],
        *,
        language: str = "zh",
        chunk_sec: float = 1.0,
        min_utterance_sec: float = 5.0,
        max_utterance_sec: float = 14.0,
        vad_silence_rms: float = 200.0,
        vad_silence_sec: float = 0.5,
        hotwords: Optional[List[str]] = None,
        context: str = "",
        include_partial: bool = True,
    ) -> Iterator[StreamEvent]:
        src_name = str(source) if isinstance(source, (str, os.PathLike)) else "<stream>"
        with self.create_stream(
            language=language,
            chunk_sec=chunk_sec,
            min_utterance_sec=min_utterance_sec,
            max_utterance_sec=max_utterance_sec,
            vad_silence_rms=vad_silence_rms,
            vad_silence_sec=vad_silence_sec,
            hotwords=hotwords,
            context=context,
            organize=False,
            source_name=src_name,
        ) as stream:
            read_size = max(6400, (int(round(chunk_sec * 16000)) * 2 // 640) * 640)
            for pcm_chunk in self.iter_pcm16k_chunks(source, chunk_bytes=read_size):
                for ev in stream.feed(pcm_chunk):
                    if include_partial or ev.is_final:
                        yield ev
            for ev in stream.flush():
                if include_partial or ev.is_final:
                    yield ev

    async def atranscribe_stream(
        self,
        source: Union[str, os.PathLike, bytes, Iterable[bytes]],
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_running_loop()
        gen = self.transcribe_stream(source, **kwargs)
        sentinel = object()
        while True:
            ev = await loop.run_in_executor(None, next, gen, sentinel)
            if ev is sentinel:
                break
            yield ev  # type: ignore[misc]

    # -------------------- 三阶段全并发流水线转写 --------------------
    def transcribe(
        self,
        source: Union[str, os.PathLike, bytes],
        *,
        mode: str = "omni",
        engines: Optional[List[str]] = None,
        engine_workers: Optional[Dict[str, int]] = None,
        language: str = "zh",
        min_segment_sec: float = 8.0,
        max_segment_sec: float = 14.0,
        silence_rms_threshold: float = 15.0,
        split: bool = True,
        hotwords: Optional[List[str]] = None,
        context: str = "",
        organize: bool = True,
        organize_workers: int = 12,
        organize_chunk_chars: int = 380,
        enable_hedging: bool = True,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> TranscriptResult:
        """
        将任意时长音频/视频文件或 PCM 字节流转写为 `TranscriptResult`。
        """
        t0 = time.perf_counter()
        if not split:
            pcm = self.load_pcm16k(source)
            dur = len(pcm) / 32000.0
            seg_iter: Iterable[tuple[int, float, float, bytes, float]] = (
                [(1, 0.0, round(dur, 2), pcm, compute_rms(pcm))] if pcm else []
            )
        else:
            seg_iter = iter_vad_segments_streaming(
                source,
                min_segment_sec=min_segment_sec,
                max_segment_sec=max_segment_sec,
                ffmpeg_path=self.ffmpeg_path,
            )

        segments, org_text, asr_sec, org_tail_sec, metrics_dict = self.scheduler.run_pipeline(
            seg_iter,
            mode=mode,
            engines=engines,
            engine_workers=engine_workers,
            language=language,
            silence_rms_threshold=silence_rms_threshold,
            hotwords=hotwords,
            context=context,
            organize=organize,
            organize_workers=organize_workers,
            organize_chunk_chars=organize_chunk_chars,
            enable_hedging=enable_hedging,
            on_segment=on_segment,
        )

        total_dur = segments[-1].end if segments else 0.0
        elapsed = time.perf_counter() - t0
        return TranscriptResult(
            segments=segments,
            organized_text=org_text,
            duration=round(total_dur, 2),
            source=str(source) if isinstance(source, (str, os.PathLike)) else "<audio_bytes>",
            mode=mode,
            language=language,
            elapsed_sec=round(elapsed, 2),
            asr_sec=asr_sec,
            organize_tail_sec=org_tail_sec,
            engine_metrics=metrics_dict,
        )


_DEFAULT_CLIENT: Optional[OmniASR] = None


def _get_default_client() -> OmniASR:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = OmniASR()
    return _DEFAULT_CLIENT


def register_engine(
    engine_or_name: Union[str, ASREngineProtocol, BaseASREngine],
    fn: Optional[Callable[..., str]] = None,
    **kwargs,
) -> ASREngineProtocol:
    """向全局默认 `OmniASR` 客户端注册一个自定义 ASR 引擎插件"""
    return _get_default_client().register_engine(engine_or_name, fn, **kwargs)


def transcribe(source: Union[str, os.PathLike, bytes], **kwargs) -> TranscriptResult:
    """便捷同步转写函数"""
    return _get_default_client().transcribe(source, **kwargs)


def transcribe_stream(
    source: Union[str, os.PathLike, bytes, Iterable[bytes]], **kwargs
) -> Iterator[StreamEvent]:
    """便捷同步流式生成器函数"""
    return _get_default_client().transcribe_stream(source, **kwargs)


def organize_text(text: str, **kwargs) -> str:
    """便捷豆包官方 LLM 智能整理函数"""
    return _get_default_client().organize_text(text, **kwargs)
