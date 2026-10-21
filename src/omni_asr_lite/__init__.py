# -*- coding: utf-8 -*-
"""
omni_asr_lite — 零第三方依赖、可插拔多引擎并发调度语音转文字与豆包 LLM 无锁分桶流式整理框架
（默认内置搜狗 14 语种/方言高速 ASR 引擎，开放 ASREngineProtocol 插件注册接口，原生支持 Python 3.15t GIL=False）
"""

from .audio import (
    compute_rms,
    encode_opus_packets,
    encode_sogou_ws_frames,
    iter_pcm16k_chunks,
    iter_vad_segments_streaming,
    load_pcm16k,
    split_pcm_by_silence,
)
from .client import OmniASR, organize_text, register_engine, transcribe, transcribe_stream
from .engines import (
    ASREngineProtocol,
    BaseASREngine,
    CallableASREngine,
    SOGOU_LANGUAGES,
    SogouEngine,
    coerce_engine,
    normalize_sogou_language,
)
from .models import (
    DEFAULT_ENGINE_WORKERS,
    EngineMetrics,
    Segment,
    StreamEvent,
    TranscriptResult,
)
from .organizer import DoubaoOrganizer, StreamingLLMOrganizer, split_text_for_organize
from .scheduler import OmniScheduler
from .stream import OmniStreamSession

__version__ = "1.0.0"
__all__ = [
    "OmniASR",
    "OmniScheduler",
    "OmniStreamSession",
    "ASREngineProtocol",
    "BaseASREngine",
    "CallableASREngine",
    "coerce_engine",
    "SogouEngine",
    "SOGOU_LANGUAGES",
    "normalize_sogou_language",
    "DoubaoOrganizer",
    "StreamingLLMOrganizer",
    "DEFAULT_ENGINE_WORKERS",
    "register_engine",
    "transcribe",
    "transcribe_stream",
    "organize_text",
    "split_text_for_organize",
    "EngineMetrics",
    "Segment",
    "StreamEvent",
    "TranscriptResult",
    "encode_sogou_ws_frames",
    "encode_opus_packets",
    "compute_rms",
    "split_pcm_by_silence",
    "iter_vad_segments_streaming",
    "iter_pcm16k_chunks",
    "load_pcm16k",
]
