# -*- coding: utf-8 -*-
"""
omni-asr-lite 单元测试、插件扩展协议测试、429 限流熔断容灾测试与端到端实测套件
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from omni_asr_lite import (
    ASREngineProtocol,
    BaseASREngine,
    OmniASR,
    SOGOU_LANGUAGES,
    SogouEngine,
    encode_opus_packets,
    encode_sogou_ws_frames,
    normalize_sogou_language,
    transcribe,
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
    def test_languages_and_opus_encoding(self):
        self.assertEqual(len(SOGOU_LANGUAGES), 14)
        self.assertEqual(normalize_sogou_language("yue"), "zh-cmnxyue-Hans-CN")
        self.assertEqual(normalize_sogou_language("sichuan"), "zh-cmn-sichuan")

        pcm_1s = b"\x10\x00" * 16000
        self.assertGreater(len(encode_sogou_ws_frames(pcm_1s)), 500)
        self.assertEqual(len(encode_opus_packets(pcm_1s, frame_ms=20)), 50)

    def test_plugin_protocol_and_rate_limit_failover(self):
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        if not os.path.isfile(sample_wav):
            self.skipTest("Sample wav not found")

        with OmniASR() as asr:
            pcm = asr.load_pcm16k(sample_wav)
            # 1. 注册一个总是抛出 429 限流异常的函数式插件引擎 flaky_cloud
            def _flaky_fn(pcm_bytes: bytes, **kwargs) -> str:
                raise RuntimeError("HTTP 429 rate limit exceeded")

            eng_flaky = asr.register_engine("flaky_cloud", _flaky_fn, workers=2, initial_rtf=0.02)
            self.assertIsInstance(eng_flaky, ASREngineProtocol)

            # 2. 注册一个继承自 BaseASREngine 的健康自定义插件引擎 custom_mock
            sogou_eng = asr.scheduler.engines["sogou"]
            assert isinstance(sogou_eng, SogouEngine)
            eng_mock = asr.register_engine(CustomMockEngine(sogou_eng), workers=2, initial_rtf=0.04)
            self.assertIsInstance(eng_mock, ASREngineProtocol)
            self.assertIn("flaky_cloud", asr.list_engines())
            self.assertIn("custom_mock", asr.list_engines())

            # 3. 仅使用 flaky_cloud + custom_mock 跑 2 段音频，验证 flaky_cloud 429 熔断后自动转移到 custom_mock
            res = asr.transcribe(
                pcm + (b"\x00\x00" * 16000) + pcm,
                engines=["flaky_cloud", "custom_mock"],
                min_segment_sec=4.0,
                max_segment_sec=6.5,
                organize=False,
            )
            self.assertEqual(len(res.lines), 2)
            for seg in res.segments:
                self.assertIn("达摩院", seg.text)
            self.assertGreaterEqual(res.engine_metrics["flaky_cloud"]["rate_limited"], 1)
            self.assertGreaterEqual(res.engine_metrics["flaky_cloud"]["circuit_trips"], 1)
            self.assertGreaterEqual(res.engine_metrics["custom_mock"]["succeeded"], 1)
            print("\n[Plugin Failover Test] 自定义插件 429 熔断与跨引擎容灾通过:", res.engine_stats)

    def test_builtin_sogou_and_streaming_llm_pipeline(self):
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        sample_wav2 = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example.wav")
        )
        if not os.path.isfile(sample_wav):
            self.skipTest("Sample wav not found")

        with OmniASR() as asr:
            pcm1 = asr.load_pcm16k(sample_wav)
            pcm2 = asr.load_pcm16k(sample_wav2)
            silence_1s = b"\x00\x00" * 16000
            long_pcm = (pcm1 + silence_1s + pcm2 + silence_1s) * 2

            res = asr.transcribe(
                long_pcm,
                min_segment_sec=4.0,
                max_segment_sec=6.5,
                organize=True,
            )
            print(
                f"\n[Omni-ASR-Lite Benchmark] {res.duration:.1f}s 音频 ({len(res.segments)} 段) "
                f"总耗时: {res.elapsed_sec:.2f}s (ASR {res.asr_sec:.2f}s + LLM收尾 {res.organize_tail_sec:.2f}s)"
            )
            self.assertEqual(len(res.lines), len(res.segments))
            self.assertIn("达摩院", res.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
