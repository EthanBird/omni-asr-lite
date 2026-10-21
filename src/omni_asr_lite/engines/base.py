# -*- coding: utf-8 -*-
"""
可扩展 ASR 引擎插件协议与基类 (`omni_asr_lite.engines.base`)：
任何外部或自定义语音识别服务，只要实现 `ASREngineProtocol`（或继承 `BaseASREngine`，或传入一个满足签名的可调用函数），
即可通过 `asr.register_engine(...)` 无缝接入 `OmniScheduler`，自动参与：
1. 多线程动态工作窃取并发调度 (Work-Stealing Multi-Threading)
2. 429 限流/异常自适应指数退避熔断 (Adaptive Circuit Breaker)
3. 跨引擎零感知自动故障转移 (Cross-Engine Auto-Failover)
4. 尾声动态准入控制与 `0.30s` 长尾对冲冲刺 (Tail Admission Control & Straggler Hedging)
5. 同段多引擎赛马择优模式 (`mode="ensemble"`)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Protocol, Set, Union, runtime_checkable


@runtime_checkable
class ASREngineProtocol(Protocol):
    """ASR 引擎插件协议规范（Duck-Typing / Runtime Checkable）"""

    name: str

    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str = "zh",
        hotwords: Optional[List[str]] = None,
        context: str = "",
        timeout: Optional[float] = None,
    ) -> str:
        """
        将一段 16kHz / 16-bit / 单声道 (s16le) PCM 字节流识别为文本字符串。
        若遇到限流 (如 HTTP 429) 或网络故障，应抛出异常以便调度器触发熔断与跨引擎故障转移。
        """
        ...

    def supports_language(self, language: str) -> bool:
        """返回该引擎是否支持目标语种/方言"""
        ...

    def prewarm(self, count: int = 4) -> None:
        """可选：流水线启动时异步预热底层连接池"""
        ...

    def close(self) -> None:
        """可选：释放底层网络连接或句柄"""
        ...


class BaseASREngine(ABC):
    """自定义 ASR 引擎抽象基类"""

    name: str = "custom"
    supported_languages: Optional[Set[str]] = None  # None 表示支持全部语种或由内部自行映射

    @abstractmethod
    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str = "zh",
        hotwords: Optional[List[str]] = None,
        context: str = "",
        timeout: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    def supports_language(self, language: str) -> bool:
        if not self.supported_languages:
            return True
        norm = (language or "zh").strip().lower().replace("_", "-")
        return norm in self.supported_languages or "*" in self.supported_languages

    def prewarm(self, count: int = 4) -> None:
        _ = count

    def close(self) -> None:
        pass


class CallableASREngine(BaseASREngine):
    """
    函数式 ASR 引擎适配器：
    允许用户直接将一个普通 Python 函数 `fn(pcm_bytes, *, language="zh", hotwords=None, context="") -> str`
    或 `fn(pcm_bytes) -> str` 注册为完整的多线程可调度引擎。
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., str],
        *,
        supported_languages: Optional[Iterable[str]] = None,
        prewarm_fn: Optional[Callable[[int], None]] = None,
        close_fn: Optional[Callable[[], None]] = None,
    ):
        self.name = str(name).strip().lower()
        self._fn = fn
        self.supported_languages = (
            {s.strip().lower().replace("_", "-") for s in supported_languages}
            if supported_languages is not None
            else None
        )
        self._prewarm_fn = prewarm_fn
        self._close_fn = close_fn

    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str = "zh",
        hotwords: Optional[List[str]] = None,
        context: str = "",
        timeout: Optional[float] = None,
    ) -> str:
        try:
            return self._fn(
                pcm_bytes,
                language=language,
                hotwords=hotwords,
                context=context,
                timeout=timeout,
            )
        except TypeError:
            # 兼容仅接收单个 pcm_bytes 参数的极简函数 fn(pcm_bytes) -> str
            return self._fn(pcm_bytes)

    def prewarm(self, count: int = 4) -> None:
        if self._prewarm_fn is not None:
            self._prewarm_fn(count)

    def close(self) -> None:
        if self._close_fn is not None:
            self._close_fn()


def coerce_engine(
    engine_or_name: Union[str, ASREngineProtocol, BaseASREngine],
    fn: Optional[Callable[..., str]] = None,
    *,
    supported_languages: Optional[Iterable[str]] = None,
) -> ASREngineProtocol:
    """将引擎对象或 `(name, callable)` 规范化为满足 `ASREngineProtocol` 的实例"""
    if isinstance(engine_or_name, str):
        if fn is None or not callable(fn):
            raise ValueError("以名称字符串注册引擎时，必须提供可调用的识别函数 fn")
        return CallableASREngine(
            engine_or_name,
            fn,
            supported_languages=supported_languages,
        )
    if not hasattr(engine_or_name, "name") or not hasattr(engine_or_name, "transcribe_pcm"):
        raise TypeError(
            f"引擎对象 {engine_or_name!r} 必须实现 ASREngineProtocol (包含 .name 属性与 .transcribe_pcm() 方法)"
        )
    return engine_or_name  # type: ignore[return-value]
