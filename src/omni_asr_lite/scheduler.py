# -*- coding: utf-8 -*-
"""
可插拔多引擎动态抢占、限流自适应熔断与长尾对冲冲刺调度器 (`omni_asr_lite.scheduler`)：
1. 默认内置 `SogouEngine` (`sogou`，16 线程并发，支持 14 种语言/方言)。
2. 开放 `register_engine(engine_or_name, fn=None, workers=8, initial_rtf=0.05)` 插件扩展机制：
   任何满足 `ASREngineProtocol` 的外部/自定义引擎接入后，自动与内置引擎组成异构并发池，共同享有：
   - 交错工作窃取调度 (Round-Robin Interleaved Work-Stealing)
   - 429 限流/异常自适应指数退避熔断 (Circuit Breaker)
   - 跨引擎自动故障转移 (Auto-Failover)
   - 相对第一梯队尾声准入控制 (Relative Fast-Tier Tail Admission Control) 与 `0.30s` 长尾对冲冲刺 (Straggler Hedging)
   - 同段多引擎赛马择优 (`mode="ensemble"`)
"""

from __future__ import annotations

from dataclasses import asdict
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from .engines import ASREngineProtocol, BaseASREngine, SogouEngine, coerce_engine
from .models import DEFAULT_ENGINE_WORKERS, EngineMetrics, Segment
from .organizer import DoubaoOrganizer, StreamingLLMOrganizer


class _SegmentTaskState:
    """跟踪单个分片的执行、跨引擎容灾与长尾对冲冲刺状态"""

    __slots__ = (
        "idx",
        "start_sec",
        "end_sec",
        "pcm_bytes",
        "rms",
        "lock",
        "done",
        "started_at",
        "active_engines",
        "tried_engines",
        "failover_chain",
        "hedged",
        "result_seg",
    )

    def __init__(self, idx: int, start_sec: float, end_sec: float, pcm_bytes: bytes, rms: float):
        self.idx = idx
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.pcm_bytes = pcm_bytes
        self.rms = rms
        self.lock = threading.Lock()
        self.done = False
        self.started_at = 0.0
        self.active_engines: Set[str] = set()
        self.tried_engines: Set[str] = set()
        self.failover_chain: List[str] = []
        self.hedged = False
        self.result_seg: Optional[Segment] = None


class OmniScheduler:
    """可插拔多引擎并发、自适应熔断容灾与无锁分桶 LLM 流水线调度器"""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._reg_lock = threading.RLock()
        self.engines: Dict[str, ASREngineProtocol] = {
            "sogou": SogouEngine(timeout=timeout),
        }
        self.engine_workers: Dict[str, int] = dict(DEFAULT_ENGINE_WORKERS)
        self.organizer = DoubaoOrganizer(timeout=45.0)

        self._metrics_lock = threading.Lock()
        self._cooldown_until: Dict[str, float] = {"sogou": 0.0}
        self._consec_failures: Dict[str, int] = {"sogou": 0}
        self._ewma_rtf: Dict[str, float] = {"sogou": 0.030}

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
        注册一个符合 `ASREngineProtocol` 的自定义 ASR 引擎实例或函数：
        - 方式 A (传入类实例): `scheduler.register_engine(MyCustomEngine(), workers=8)`
        - 方式 B (传入函数): `scheduler.register_engine("my_cloud", my_transcribe_fn, workers=8)`
        """
        eng = coerce_engine(engine_or_name, fn, supported_languages=supported_languages)
        name = eng.name.strip().lower()
        if not name:
            raise ValueError("引擎名称 name 不能为空")
        with self._reg_lock, self._metrics_lock:
            self.engines[name] = eng
            self.engine_workers[name] = max(1, int(workers))
            self._cooldown_until[name] = 0.0
            self._consec_failures[name] = 0
            self._ewma_rtf[name] = max(0.005, float(initial_rtf))
        return eng

    def unregister_engine(self, name: str) -> None:
        key = name.strip().lower()
        with self._reg_lock, self._metrics_lock:
            eng = self.engines.pop(key, None)
            self.engine_workers.pop(key, None)
            self._cooldown_until.pop(key, None)
            self._consec_failures.pop(key, None)
            self._ewma_rtf.pop(key, None)
        if eng is not None and hasattr(eng, "close"):
            try:
                eng.close()
            except Exception:
                pass

    def list_engines(self) -> List[str]:
        with self._reg_lock:
            return list(self.engines.keys())

    def select_engines_for_language(
        self,
        language: str = "zh",
        requested_engines: Optional[List[str]] = None,
    ) -> List[str]:
        with self._reg_lock:
            reg_keys = list(self.engines.keys())
            if requested_engines:
                valid = [e.strip().lower() for e in requested_engines if e.strip().lower() in self.engines]
                if valid:
                    return valid
            matched = []
            for k in reg_keys:
                eng = self.engines[k]
                if hasattr(eng, "supports_language"):
                    try:
                        if eng.supports_language(language):
                            matched.append(k)
                        continue
                    except Exception:
                        pass
                matched.append(k)
            return matched if matched else reg_keys

    def _is_engine_healthy(self, name: str) -> bool:
        with self._metrics_lock:
            return time.perf_counter() >= self._cooldown_until.get(name, 0.0)

    def _record_success(
        self,
        metrics_map: Dict[str, EngineMetrics],
        engine_name: str,
        audio_sec: float,
        latency_sec: float,
        is_hedged_win: bool = False,
    ) -> None:
        with self._metrics_lock:
            self._consec_failures[engine_name] = 0
            self._cooldown_until[engine_name] = 0.0
            rtf = latency_sec / max(0.5, audio_sec)
            prev = self._ewma_rtf.get(engine_name, rtf)
            new_ewma = 0.7 * prev + 0.3 * rtf
            self._ewma_rtf[engine_name] = new_ewma

            m = metrics_map[engine_name]
            m.succeeded += 1
            if is_hedged_win:
                m.hedged_wins += 1
            m.total_audio_sec += audio_sec
            m.total_latency_sec += latency_sec
            m.ewma_rtf = round(new_ewma, 4)

    def _record_failure(
        self,
        metrics_map: Dict[str, EngineMetrics],
        engine_name: str,
        err: Optional[Exception] = None,
    ) -> None:
        err_str = str(err or "").lower()
        is_rate_limit = any(
            k in err_str
            for k in ("40200011", "quota", "rate", "limit", "429", "503", "refused", "rejected")
        )
        with self._metrics_lock:
            m = metrics_map[engine_name]
            m.failed += 1
            self._consec_failures[engine_name] = self._consec_failures.get(engine_name, 0) + 1
            cf = self._consec_failures[engine_name]
            if is_rate_limit:
                m.rate_limited += 1
            if is_rate_limit or cf >= 2:
                cooldown = min(6.0, 1.2 * (1.6 ** min(cf, 4)))
                self._cooldown_until[engine_name] = time.perf_counter() + cooldown
                m.circuit_trips += 1

    def _pick_failover_engine(
        self,
        active_engines: List[str],
        tried: Set[str],
    ) -> str:
        now = time.perf_counter()
        with self._metrics_lock:
            untried_healthy = [
                e for e in active_engines
                if e not in tried and now >= self._cooldown_until.get(e, 0.0)
            ]
            if untried_healthy:
                return min(untried_healthy, key=lambda e: self._ewma_rtf.get(e, 0.1))
            untried = [e for e in active_engines if e not in tried]
            if untried:
                return min(untried, key=lambda e: self._ewma_rtf.get(e, 0.1))
            return min(active_engines, key=lambda e: self._ewma_rtf.get(e, 0.1))

    def run_pipeline(
        self,
        segment_iter: Iterable[Tuple[int, float, float, bytes, float]],
        *,
        mode: str = "omni",
        engines: Optional[List[str]] = None,
        engine_workers: Optional[Dict[str, int]] = None,
        language: str = "zh",
        silence_rms_threshold: float = 15.0,
        hotwords: Optional[List[str]] = None,
        context: str = "",
        organize: bool = True,
        organize_workers: int = 12,
        organize_chunk_chars: int = 380,
        enable_hedging: bool = True,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> Tuple[List[Segment], str, float, float, Dict[str, Dict[str, Any]]]:
        t_start = time.perf_counter()
        mode_norm = (mode or "omni").strip().lower()
        with self._reg_lock:
            if mode_norm in self.engines:
                active_engines = [mode_norm]
            else:
                active_engines = self.select_engines_for_language(language, engines)
            workers_cfg = dict(self.engine_workers)

        if engine_workers:
            for k, v in engine_workers.items():
                kl = k.strip().lower()
                if kl in self.engines and int(v) >= 1:
                    workers_cfg[kl] = int(v)

        metrics_map: Dict[str, EngineMetrics] = {
            e: EngineMetrics(name=e, workers=workers_cfg.get(e, 8))
            for e in active_engines
        }

        for eng_name in active_engines:
            eng_obj = self.engines.get(eng_name)
            if eng_obj is not None and hasattr(eng_obj, "prewarm"):
                try:
                    eng_obj.prewarm(min(12, workers_cfg.get(eng_name, 8)))
                except Exception:
                    pass
        if organize:
            self.organizer.prewarm(min(8, organize_workers))

        llm_pipeline: Optional[StreamingLLMOrganizer] = (
            self.organizer.create_streaming_pipeline(
                chunk_chars=organize_chunk_chars, workers=organize_workers
            )
            if organize
            else None
        )

        # ==================== 模式 B: Ensemble 同段多引擎赛马择优 ====================
        if mode_norm == "ensemble":
            all_raw = list(segment_iter)
            results_ens: List[Segment] = []

            def _run_ensemble_seg(item: Tuple[int, float, float, bytes, float]) -> Segment:
                idx, s_sec, e_sec, sub_pcm, rms = item
                dur = max(0.1, e_sec - s_sec)
                if rms < silence_rms_threshold:
                    seg = Segment(index=idx, start=s_sec, end=e_sec, text="", rms=round(rms, 1), engine="silent")
                else:
                    race_engines = active_engines[:3]
                    cands: Dict[str, str] = {}
                    t_seg0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=len(race_engines)) as sub_ex:
                        fut_map = {
                            sub_ex.submit(
                                self.engines[eng].transcribe_pcm,
                                sub_pcm,
                                language=language,
                                hotwords=hotwords,
                                context=context,
                            ): eng
                            for eng in race_engines
                        }
                        for fut in as_completed(fut_map):
                            eng = fut_map[fut]
                            try:
                                txt = (fut.result() or "").strip()
                                if txt:
                                    cands[eng] = txt
                                    self._record_success(metrics_map, eng, dur, time.perf_counter() - t_seg0)
                            except Exception as exc:
                                self._record_failure(metrics_map, eng, exc)
                    if cands:
                        best_eng = max(cands, key=lambda k: (len(cands[k].strip("，。！？,.!? ")), len(cands[k])))
                        best_txt = cands[best_eng]
                        win_label = f"ensemble({best_eng})" if len(race_engines) > 1 else best_eng
                    else:
                        best_txt = ""
                        win_label = "ensemble(none)"
                    seg = Segment(
                        index=idx,
                        start=s_sec,
                        end=e_sec,
                        text=best_txt,
                        rms=round(rms, 1),
                        engine=win_label,
                        latency_sec=round(time.perf_counter() - t_seg0, 2),
                        candidates=cands,
                    )
                if llm_pipeline is not None:
                    llm_pipeline.on_segment_ready(seg)
                if on_segment is not None:
                    on_segment(seg)
                return seg

            with ThreadPoolExecutor(max_workers=8) as ens_pool:
                futs = [ens_pool.submit(_run_ensemble_seg, it) for it in all_raw]
                results_ens = [f.result() for f in futs]

            results_ens.sort(key=lambda s: s.index)
            asr_sec = time.perf_counter() - t_start
            t_org_tail0 = time.perf_counter()
            org_text = llm_pipeline.finalize() if llm_pipeline is not None else ""
            org_tail_sec = time.perf_counter() - t_org_tail0
            return (
                results_ens,
                org_text,
                round(asr_sec, 2),
                round(org_tail_sec, 2),
                {k: {**asdict(v), "avg_speed_x": v.avg_speed_x} for k, v in metrics_map.items()},
            )

        # ==================== 模式 A (默认): Omni 动态抢占 + 熔断容灾 + 长尾冲刺 ====================
        task_queue: "queue.Queue[Optional[_SegmentTaskState]]" = queue.Queue()
        all_tasks: List[_SegmentTaskState] = []
        tasks_lock = threading.Lock()
        producer_done = threading.Event()
        completed_segments: Dict[int, Segment] = {}
        completed_lock = threading.Lock()

        def _complete_task(
            task: _SegmentTaskState,
            text: str,
            winner_engine: str,
            latency_sec: float,
            is_hedged: bool = False,
        ) -> bool:
            with task.lock:
                if task.done:
                    return False
                task.done = True
                chain = list(task.failover_chain)
                if winner_engine not in chain:
                    chain.append(winner_engine)
                seg = Segment(
                    index=task.idx,
                    start=task.start_sec,
                    end=task.end_sec,
                    text=text,
                    rms=round(task.rms, 1),
                    engine=winner_engine,
                    latency_sec=round(latency_sec, 2),
                    failover_chain=chain,
                )
                task.result_seg = seg

            dur = max(0.1, task.end_sec - task.start_sec)
            base_eng = winner_engine.split("(")[0]
            if base_eng in metrics_map:
                self._record_success(metrics_map, base_eng, dur, latency_sec, is_hedged_win=is_hedged)

            with completed_lock:
                completed_segments[task.idx] = seg
            if llm_pipeline is not None:
                llm_pipeline.on_segment_ready(seg)
            if on_segment is not None:
                on_segment(seg)
            return True

        def _execute_on_engine(task: _SegmentTaskState, eng_name: str, is_hedged: bool = False) -> None:
            with task.lock:
                if task.done:
                    return
                if not task.started_at:
                    task.started_at = time.perf_counter()
                task.active_engines.add(eng_name)
                task.tried_engines.add(eng_name)
                if not is_hedged:
                    task.failover_chain.append(eng_name)

            t0 = time.perf_counter()
            try:
                txt = (
                    self.engines[eng_name].transcribe_pcm(
                        task.pcm_bytes,
                        language=language,
                        hotwords=hotwords,
                        context=context,
                    )
                    or ""
                ).strip()
                dt = time.perf_counter() - t0
                if not txt and task.rms >= 280.0 and len(active_engines) > 1:
                    with task.lock:
                        can_retry = not task.done and len(task.tried_engines) < len(active_engines)
                    if can_retry:
                        self._record_failure(metrics_map, eng_name, RuntimeError("empty_transcript_on_voiced_segment"))
                        if not is_hedged:
                            _failover_task(task)
                        return
                _complete_task(task, txt, eng_name, dt, is_hedged=is_hedged)
            except Exception as exc:
                self._record_failure(metrics_map, eng_name, exc)
                if not is_hedged:
                    _failover_task(task)
            finally:
                with task.lock:
                    task.active_engines.discard(eng_name)

        def _failover_task(task: _SegmentTaskState) -> None:
            max_attempts = max(3, len(active_engines))
            for _ in range(max_attempts):
                with task.lock:
                    if task.done:
                        return
                    tried = set(task.tried_engines)
                backup_eng = self._pick_failover_engine(active_engines, tried)
                with task.lock:
                    if task.done:
                        return
                    task.active_engines.add(backup_eng)
                    task.tried_engines.add(backup_eng)
                    task.failover_chain.append(backup_eng)
                t_f0 = time.perf_counter()
                try:
                    txt = (
                        self.engines[backup_eng].transcribe_pcm(
                            task.pcm_bytes,
                            language=language,
                            hotwords=hotwords,
                            context=context,
                        )
                        or ""
                    ).strip()
                    dt_f = time.perf_counter() - t_f0
                    label = f"{backup_eng}(failover)" if len(active_engines) > 1 else backup_eng
                    _complete_task(task, txt, label, dt_f, is_hedged=False)
                    return
                except Exception as exc:
                    self._record_failure(metrics_map, backup_eng, exc)
                finally:
                    with task.lock:
                        task.active_engines.discard(backup_eng)

            _complete_task(task, "", "failed", 0.0, is_hedged=False)

        def _is_fast_tier(eng_name: str) -> bool:
            with self._metrics_lock:
                best_rtf = min((self._ewma_rtf.get(e, 0.05) for e in active_engines), default=0.05)
                my_rtf = self._ewma_rtf.get(eng_name, 0.05)
            return my_rtf <= max(0.09, best_rtf * 1.65)

        def _worker_loop(my_engine: str) -> None:
            while True:
                if (
                    producer_done.is_set()
                    and len(active_engines) > 1
                    and 0 < task_queue.qsize() <= 12
                    and not _is_fast_tier(my_engine)
                ):
                    task = None
                else:
                    try:
                        task = task_queue.get(timeout=0.03)
                    except queue.Empty:
                        task = None

                if task is not None:
                    try:
                        chosen_engine = my_engine
                        if not self._is_engine_healthy(my_engine) and len(active_engines) > 1:
                            with task.lock:
                                tried_snap = set(task.tried_engines)
                            chosen_engine = self._pick_failover_engine(active_engines, tried_snap)
                        _execute_on_engine(task, chosen_engine, is_hedged=False)
                    finally:
                        task_queue.task_done()
                    continue

                if producer_done.is_set():
                    with tasks_lock:
                        unfinished = [t for t in all_tasks if not t.done]
                    if not unfinished:
                        break

                    if (
                        enable_hedging
                        and len(active_engines) > 1
                        and self._is_engine_healthy(my_engine)
                        and _is_fast_tier(my_engine)
                    ):
                        now = time.perf_counter()
                        straggler: Optional[_SegmentTaskState] = None
                        for t in unfinished:
                            with t.lock:
                                elapsed_t = (now - t.started_at) if t.started_at > 0 else 0.0
                                if not t.done and elapsed_t >= 18.0:
                                    straggler = None
                                    break
                                if (
                                    not t.done
                                    and (not t.hedged or elapsed_t >= 6.0)
                                    and elapsed_t >= 0.30
                                    and my_engine not in t.active_engines
                                ):
                                    t.hedged = True
                                    straggler = t
                                    break
                        else:
                            straggler = None
                        for t in unfinished:
                            if not t.done and t.started_at > 0 and (now - t.started_at) >= 18.0:
                                _complete_task(t, "", "timeout", now - t.started_at, is_hedged=False)
                        if straggler is not None:
                            _execute_on_engine(straggler, my_engine, is_hedged=True)
                            continue
                    time.sleep(0.01)

        worker_threads: List[threading.Thread] = []
        max_w = max((workers_cfg.get(e, 8) for e in active_engines), default=1)
        for round_idx in range(max_w):
            for eng in active_engines:
                if round_idx < max(1, workers_cfg.get(eng, 8)):
                    th = threading.Thread(
                        target=_worker_loop,
                        args=(eng,),
                        name=f"omni-lite-{eng}-{round_idx}",
                        daemon=True,
                    )
                    th.start()
                    worker_threads.append(th)

        try:
            for idx, s_sec, e_sec, sub_pcm, rms in segment_iter:
                t_state = _SegmentTaskState(idx, s_sec, e_sec, sub_pcm, rms)
                if rms < silence_rms_threshold:
                    _complete_task(t_state, "", "silent", 0.0)
                    with tasks_lock:
                        all_tasks.append(t_state)
                else:
                    with tasks_lock:
                        all_tasks.append(t_state)
                    task_queue.put(t_state)
        finally:
            producer_done.set()

        for th in worker_threads:
            th.join()

        with completed_lock:
            ordered_segments = [completed_segments[k] for k in sorted(completed_segments)]

        asr_sec = time.perf_counter() - t_start
        t_org_tail0 = time.perf_counter()
        org_text = llm_pipeline.finalize() if llm_pipeline is not None else ""
        org_tail_sec = time.perf_counter() - t_org_tail0

        metrics_dict = {
            k: {**asdict(v), "avg_speed_x": v.avg_speed_x}
            for k, v in metrics_map.items()
        }
        return (
            ordered_segments,
            org_text,
            round(asr_sec, 2),
            round(org_tail_sec, 2),
            metrics_dict,
        )

    def close(self) -> None:
        with self._reg_lock:
            engs = list(self.engines.values())
        for eng in engs:
            if hasattr(eng, "close"):
                try:
                    eng.close()
                except Exception:
                    pass
        self.organizer.close()
