# -*- coding: utf-8 -*-
"""
豆包官方免鉴权 LLM「智能整理」与无队头阻塞分桶流式整理器 (`omni_asr_lite.organizer`)：
1. 针对豆包服务端偶发单连接长尾卡顿（如 ~18s 慢响应），实装 **1.6s 投机对冲重试 (Speculative LLM Hedging)** +
   **6.0s 硬超时自动降级回原文**：当主请求超过 1.6s 未返回时，立即在新建连接上并发发起备份对冲请求，谁先返回用谁！
   （>200 字的长文本按长度等比放大对冲时刻，避免高并发下对正常长请求的无谓重复；主请求提前失败则立即换连接重试。）
2. `StreamingLLMOrganizer`：按 7 段定长分桶无队头阻塞并发提交，ASR 进入尾声后凑齐的桶与收尾桶改用 220 字微块极速收口。
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .http_pool import HTTPSConnectionPool
from .models import Segment

DOUBAO_ORGANIZE_HOST = "ime.oceancloudapi.com"
DOUBAO_ORGANIZE_PATH = "/api/v2/ai/text_organization"
DOUBAO_ORGANIZE_SCENE = 6
DOUBAO_ORGANIZE_UA = (
    "com.bytedance.android.doubaoime/100502008 "
    "(Linux; U; Android 16; zh_CN; Pixel 7 Pro; Build/BP2A.250605.031.A2)"
)
_PUNC_END = tuple("。！？!?；;，,、：:")


def split_text_for_organize(text: str, max_chars: int = 420) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    seps = "。！？!?\n；;，,"
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        if start + max_chars >= n:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break
        end = start + max_chars
        split_pos = -1
        for sep in seps:
            pos = text.rfind(sep, start + max_chars // 2, end)
            if pos != -1:
                split_pos = pos + 1
                break
        if split_pos == -1:
            split_pos = end
        piece = text[start:split_pos].strip()
        if piece:
            chunks.append(piece)
        start = split_pos
    return chunks


def _join_seg_texts(texts: List[str]) -> str:
    parts: List[str] = []
    for raw in texts:
        t = raw.strip()
        if not t:
            continue
        if parts and not parts[-1].endswith(_PUNC_END):
            parts[-1] += "，"
        parts.append(t)
    if parts and not parts[-1].endswith(_PUNC_END):
        parts[-1] += "。"
    return "".join(parts)


class DoubaoOrganizer:
    """豆包官方免鉴权 LLM「智能整理」并发客户端（带 1.6s 投机对冲与 6.0s 硬超时降级保护）"""

    def __init__(self, timeout: float = 6.0, hedge_delay: float = 1.6):
        self.timeout = timeout
        self.hedge_delay = hedge_delay
        self.pool = HTTPSConnectionPool(DOUBAO_ORGANIZE_HOST, timeout=timeout, max_pool_size=24)
        self._warm_lock = threading.Lock()
        self._warming = 0

    def prewarm(self, count: int = 8) -> None:
        """后台预建 Keep-Alive 连接，按「目标数 - 池中空闲数 - 正在预热数」差额补齐，重复调用不会超建"""

        def _warm_one():
            try:
                conn = self.pool._create_conn(timeout=5.0)
                self.pool._release(conn)
            except Exception:
                pass
            finally:
                with self._warm_lock:
                    self._warming -= 1

        with self._warm_lock:
            deficit = max(0, int(count)) - self.pool.idle_count() - self._warming
            deficit = max(0, deficit)
            self._warming += deficit
        for _ in range(deficit):
            threading.Thread(target=_warm_one, daemon=True).start()

    def _call_once(self, body: bytes, timeout: float, fresh: bool = False) -> Optional[str]:
        try:
            status, raw_bytes = self.pool.post(
                DOUBAO_ORGANIZE_PATH,
                body,
                headers={"Content-Type": "application/json", "User-Agent": DOUBAO_ORGANIZE_UA},
                timeout=timeout,
                fresh=fresh,
            )
            if status == 200 and raw_bytes:
                d = json.loads(raw_bytes.decode("utf-8", "ignore"))
                if d.get("code") == 0:
                    content = (d.get("data") or {}).get("content")
                    if content:
                        return str(content)
        except Exception:
            pass
        return None

    def organize_single(
        self,
        text: str,
        *,
        space_cn_en: int = 1,
        space_newline: int = 0,
        timeout: Optional[float] = None,
        hedge_delay: Optional[float] = None,
    ) -> str:
        """
        整理单个文本块：
        - 主请求若在对冲时刻内返回，直接零额外开销返回；对冲时刻默认取 `self.hedge_delay` (1.6s)，
          文本超过 200 字时按长度等比放大（如 380 字 ≈ 3.0s）——LLM 生成耗时随文本长度增长，实测在高并发下
          固定 1.6s 会对大量「正常但较长」的请求发起无谓的重复请求，反而拖慢服务端整体吞吐；
        - 若主请求超过该时长仍未响应（遇到偶发服务端慢连接），立即在新建 TLS 连接上并发发起备份对冲请求，
          主备任意一方率先成功即刻唤醒返回，总等待上限严格不超过 `timeout`，超时平稳降级回原文本！
        - 若主请求在对冲时刻之前就已失败（如网络错误 / 非 200），立即在新连接上重试，而不是空等到对冲时刻；
          主备均失败时立即降级返回原文，绝不空等至 `timeout`。
        """
        text = text.strip()
        if not text:
            return ""
        to = timeout or self.timeout
        if hedge_delay is None:
            hd = self.hedge_delay * max(1.0, len(text) / 200.0)
        else:
            hd = max(0.0, float(hedge_delay))
        body = json.dumps(
            {
                "scene": DOUBAO_ORGANIZE_SCENE,
                "query": text,
                "space_at_cn_en_nb": space_cn_en,
                "space_at_newline": space_newline,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        cond = threading.Condition()
        winner: List[str] = []
        state = {"launched": 0, "finished": 0}

        def _worker(fresh_conn: bool):
            res = self._call_once(body, timeout=to, fresh=fresh_conn)
            with cond:
                state["finished"] += 1
                if res and not winner:
                    winner.append(res)
                cond.notify_all()

        def _launch(fresh_conn: bool) -> None:
            state["launched"] += 1
            threading.Thread(target=_worker, args=(fresh_conn,), daemon=True).start()

        t_start = time.monotonic()
        deadline = t_start + to
        hedge_at = t_start + min(hd, to)
        with cond:
            _launch(False)
            while not winner:
                now = time.monotonic()
                if now >= deadline:
                    break
                all_failed = state["finished"] >= state["launched"]
                if state["launched"] < 2 and (now >= hedge_at or all_failed):
                    # 主请求超过对冲时刻未返回 / 已提前失败：启动新建连接备份请求，消除偶发长尾卡顿
                    _launch(True)
                    continue
                if all_failed:
                    break
                wake_at = hedge_at if state["launched"] < 2 else deadline
                cond.wait(timeout=max(0.001, min(wake_at, deadline) - now))
            return winner[0] if winner else text

    def organize_text(
        self,
        text: str,
        *,
        max_chunk_chars: int = 420,
        workers: int = 10,
        timeout: float = 6.0,
    ) -> str:
        text = text.strip()
        if not text:
            return ""
        chunks = split_text_for_organize(text, max_chars=max_chunk_chars)
        if len(chunks) == 1:
            return self.organize_single(chunks[0], timeout=timeout)

        actual_w = max(1, min(int(workers), len(chunks), 16))
        with ThreadPoolExecutor(max_workers=actual_w) as ex:
            futures = [ex.submit(self.organize_single, c, timeout=timeout) for c in chunks]
            parts = [f.result().strip() for f in futures]
        return "\n\n".join(p for p in parts if p)

    def create_streaming_pipeline(
        self,
        *,
        chunk_chars: int = 380,
        workers: int = 12,
        bucket_segments: int = 7,
    ) -> "StreamingLLMOrganizer":
        return StreamingLLMOrganizer(
            self,
            chunk_chars=chunk_chars,
            workers=workers,
            bucket_segments=bucket_segments,
        )

    def close(self) -> None:
        self.pool.close_all()


class StreamingLLMOrganizer:
    """
    无队头阻塞分桶流式 LLM 整理器：
    按每 `bucket_segments` (默认 7 段，约 320 字) 划分为独立桶，
    任意桶凑齐即刻异步发往豆包 LLM，收尾未满桶自动按 220 字微块并发冲刺。
    """

    def __init__(
        self,
        organizer: DoubaoOrganizer,
        chunk_chars: int = 380,
        workers: int = 12,
        bucket_segments: int = 7,
        tail_chunk_chars: int = 220,
    ):
        self.organizer = organizer
        self.chunk_chars = max(180, int(chunk_chars))
        self.tail_chunk_chars = max(120, min(self.chunk_chars, int(tail_chunk_chars)))
        self.bucket_segments = max(3, int(bucket_segments))
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._lock = threading.Lock()
        self._buckets: Dict[int, Dict[int, str]] = {}
        self._submitted_buckets: Dict[int, List[Future]] = {}
        self._tail_mode = False

    def enter_tail_mode(self) -> None:
        """
        通知流水线 ASR 已进入尾声（所有分片均已派发，仅剩在途请求）：此后凑齐的桶处于端到端关键路径上，
        改用更小的 `tail_chunk_chars` 微块并发整理（实测 LLM 耗时随文本长度增长：~220 字 ≈0.9s，~380 字 ≈1.0~1.9s）。
        """
        self._tail_mode = True

    def _submit_bucket_locked(self, b_id: int, max_chars: int) -> None:
        seg_map = self._buckets.pop(b_id, None)
        if not seg_map:
            return
        ordered_texts = [seg_map[k] for k in sorted(seg_map) if seg_map[k].strip()]
        joined = _join_seg_texts(ordered_texts)
        if not joined.strip():
            self._submitted_buckets[b_id] = []
            return
        chunks = split_text_for_organize(joined, max_chars=max_chars)
        self._submitted_buckets[b_id] = [
            self._executor.submit(self.organizer.organize_single, c)
            for c in chunks
            if c.strip()
        ]

    def on_segment_ready(self, seg: Segment) -> None:
        b_id = (max(1, seg.index) - 1) // self.bucket_segments
        with self._lock:
            if b_id in self._submitted_buckets:
                return
            b_map = self._buckets.setdefault(b_id, {})
            b_map[seg.index] = seg.text.strip()
            if len(b_map) >= self.bucket_segments:
                max_chars = self.tail_chunk_chars if self._tail_mode else self.chunk_chars
                self._submit_bucket_locked(b_id, max_chars=max_chars)

    def finalize(self) -> str:
        with self._lock:
            remaining_b_ids = sorted(self._buckets.keys())
            for b_id in remaining_b_ids:
                self._submit_bucket_locked(b_id, max_chars=self.tail_chunk_chars)
            ordered_b_ids = sorted(self._submitted_buckets.keys())
            all_fut_groups: List[Tuple[int, List[Future]]] = [
                (b_id, self._submitted_buckets[b_id]) for b_id in ordered_b_ids
            ]

        try:
            parts: List[str] = []
            for _, futs in all_fut_groups:
                for f in futs:
                    res = f.result().strip()
                    if res:
                        parts.append(res)
            return "\n\n".join(parts)
        finally:
            self._executor.shutdown(wait=False)
