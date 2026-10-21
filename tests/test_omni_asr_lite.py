# -*- coding: utf-8 -*-
"""
omni-asr-lite 单元测试、边界防护测试、插件扩展协议测试、429 限流熔断容灾测试与真增量流式实测套件
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import omni_asr_lite.audio as audio_mod
from omni_asr_lite import (
    ASREngineProtocol,
    BaseASREngine,
    CallableASREngine,
    DoubaoOrganizer,
    OmniASR,
    SOGOU_LANGUAGES,
    SogouEngine,
    compute_rms,
    encode_opus_packets,
    encode_sogou_ws_frames,
    ensure_libopus,
    normalize_sogou_language,
    transcribe_stream,
)


class CustomMockEngine(BaseASREngine):
    """用于测试插件扩展协议的自定义引擎"""

    name = "custom_mock"
    supported_languages = {"zh", "en"}

    def __init__(self, delegate: SogouEngine):
        self._delegate = delegate

    def transcribe_pcm(self, pcm_bytes: bytes, *, language: str = "zh", hotwords=None, context="", timeout=None) -> str:
        return self._delegate.transcribe_pcm(
            pcm_bytes, language=language, hotwords=hotwords, context=context, timeout=timeout
        )


class TestOmniASRLiteSuite(unittest.TestCase):
    def test_languages_high_bitrate_ws_framing_and_missing_libopus_guard(self):
        self.assertEqual(len(SOGOU_LANGUAGES), 14)
        self.assertEqual(normalize_sogou_language("yue"), "zh-cmnxyue-Hans-CN")
        self.assertEqual(normalize_sogou_language("sichuan"), "zh-cmn-sichuan")

        # 1. 验证常规码率 (24kbps) 与高码率 (128kbps, 单帧 payload >= 126B 触发 0xFE 16-bit 扩展帧头) 均不溢出
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        with OmniASR() as asr:
            pcm = asr.load_pcm16k(sample_wav)
        ws_24k = encode_sogou_ws_frames(pcm[:32000], bitrate=24000)
        ws_128k = encode_sogou_ws_frames(pcm[:32000], bitrate=128000)
        self.assertGreater(len(ws_24k), 1000)
        self.assertGreater(len(ws_128k), len(ws_24k))
        self.assertEqual(len(encode_opus_packets(pcm[:32000], frame_ms=20)), 50)

        # 2. 验证精确全样本 compute_rms
        self.assertGreater(compute_rms(pcm), 500.0)

        # 3. 验证当 _LIBOPUS 为 None 时立即抛出显式 RuntimeError，绝不静默返回 b""
        with patch.object(audio_mod, "_LIBOPUS", None):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_libopus()
            self.assertIn("libopus", str(ctx.exception))
            with self.assertRaises(RuntimeError):
                encode_sogou_ws_frames(pcm[:3200])

    def test_callable_engine_signature_inspection_does_not_swallow_user_type_error(self):
        calls = []

        def buggy_user_fn(pcm_bytes: bytes, *, language: str = "zh", **kwargs) -> str:
            calls.append(1)
            # 模拟用户函数内部自身代码的真实 TypeError bug
            return "text" + 123  # type: ignore[operator]

        eng = CallableASREngine("buggy", buggy_user_fn)
        self.assertTrue(eng._accepts_kwargs)
        with self.assertRaises(TypeError):
            eng.transcribe_pcm(b"\x00\x00" * 1600, language="zh")
        # 确认只调用了 1 次，没有因误判为签名不匹配而重发第 2 次请求
        self.assertEqual(len(calls), 1)

        # 同时验证单参数简写函数 fn(pcm_bytes) 也能被静态识别并正常调用
        simple_eng = CallableASREngine("simple", lambda pcm: f"len={len(pcm)}")
        self.assertFalse(simple_eng._accepts_kwargs)
        self.assertEqual(simple_eng.transcribe_pcm(b"\x00" * 640, language="zh"), "len=640")

    def test_doubao_llm_speculative_hedging_and_timeout_fallback(self):
        org = DoubaoOrganizer(timeout=2.0, hedge_delay=0.25)
        call_args = []

        def fake_call_once(body: bytes, timeout: float, fresh: bool = False):
            call_args.append(fresh)
            if not fresh:
                # 模拟主连接遭遇 18s 服务端挂起
                time.sleep(2.5)
                return "慢连接结果"
            # 备份对冲连接在 0.1s 内快速返回
            time.sleep(0.08)
            return "对冲快速整理结果。"

        with patch.object(org, "_call_once", side_effect=fake_call_once):
            t0 = time.perf_counter()
            out = org.organize_single("嗯那个测试一下对冲请求")
            dt = time.perf_counter() - t0
            self.assertEqual(out, "对冲快速整理结果。")
            self.assertLess(dt, 0.8)
            self.assertEqual(call_args, [False, True])

    def test_plugin_protocol_rate_limit_failover_and_strict_mode(self):
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        with OmniASR() as asr:
            pcm = asr.load_pcm16k(sample_wav)
            asr.register_engine(
                "flaky_cloud",
                lambda pcm_bytes, **kw: (_ for _ in ()).throw(RuntimeError("HTTP 429 rate limit exceeded")),
                workers=2,
                initial_rtf=0.02,
            )
            sogou_eng = asr.scheduler.engines["sogou"]
            assert isinstance(sogou_eng, SogouEngine)
            asr.register_engine(CustomMockEngine(sogou_eng), workers=2, initial_rtf=0.04)

            res = asr.transcribe(
                pcm + (b"\x00\x00" * 16000) + pcm,
                engines=["flaky_cloud", "custom_mock"],
                min_segment_sec=4.0,
                max_segment_sec=6.5,
                organize=False,
                strict=True,
            )
            self.assertEqual(res.failed_segments, [])
            self.assertEqual(len(res.lines), 2)
            self.assertGreaterEqual(res.engine_metrics["flaky_cloud"]["rate_limited"], 1)
            self.assertGreaterEqual(res.engine_metrics["custom_mock"]["succeeded"], 1)

            # 验证当所有引擎均失败时，默认发出 UserWarning 且 strict=True 抛出 RuntimeError
            with warnings.catch_warnings(record=True) as w_rec:
                warnings.simplefilter("always")
                fail_res = asr.transcribe(
                    pcm,
                    engines=["flaky_cloud"],
                    split=False,
                    organize=False,
                    strict=False,
                )
                self.assertEqual(fail_res.failed_segments, [1])
                self.assertTrue(any("OmniASR 告警" in str(w.message) for w in w_rec))

            with self.assertRaises(RuntimeError):
                asr.transcribe(pcm, engines=["flaky_cloud"], split=False, organize=False, strict=True)

    def test_true_incremental_streaming_and_full_pipeline(self):
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        events = list(transcribe_stream(sample_wav, chunk_sec=0.5))
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(events[-1].is_final)
        self.assertIn("达摩院", events[-1].text)
        print(f"\n[True Incremental Stream] 收到 {len(events)} 个流式事件，最终定稿: {events[-1].text}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
