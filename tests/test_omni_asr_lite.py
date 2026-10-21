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


class TestPerformancePaths(unittest.TestCase):
    """性能优化路径的离线正确性测试（不依赖网络）"""

    @classmethod
    def setUpClass(cls):
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        cls.pcm = audio_mod.load_pcm16k(sample_wav)

    def test_native_aes_backend_matches_pure_python_and_nist_vector(self):
        from omni_asr_lite import crypto

        pure = crypto._PurePythonAES256()
        self.assertTrue(crypto._self_test(pure))
        self.assertTrue(crypto._self_test(crypto._AES_BACKEND))
        for n in (0, 1, 15, 16, 17, 255, 1227, 4096):
            key, iv, pt = os.urandom(32), os.urandom(16), os.urandom(n)
            ct = crypto.aes_256_cbc_encrypt(key, iv, pt)
            self.assertEqual(len(ct), (n // 16 + 1) * 16)
            self.assertEqual(ct, pure.encrypt_cbc_pkcs7(key, iv, pt))

    def test_chunked_ws_frame_encoding_is_byte_identical(self):
        from omni_asr_lite.audio import SogouWSFrameEncoder, iter_sogou_ws_frames

        pcm = self.pcm[: 32000 * 3 + 1000]  # 含非 640B 对齐的尾帧
        whole = encode_sogou_ws_frames(pcm)
        chunks = list(iter_sogou_ws_frames(pcm, chunk_frames=7))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(b"".join(chunks), whole)

        enc = SogouWSFrameEncoder()
        try:
            self.assertEqual(enc.encode(pcm), whole)  # 全新编码器的首次输出与线程局部编码器一致
            self.assertGreater(len(enc.encode(pcm[:6400])), 0)  # 跨调用保持连续状态
        finally:
            enc.close()

    def test_collect_all_final_sentences_until_close_frame(self):
        import json as _json
        from omni_asr_lite.engines import sogou as sogou_mod

        def srv_frame(opcode: int, payload: bytes) -> bytes:
            n = len(payload)
            hdr = bytes([0x80 | opcode, n]) if n < 126 else bytes([0x80 | opcode, 126]) + n.to_bytes(2, "big")
            return hdr + payload

        def result(text: str) -> bytes:
            return _json.dumps(
                {"results": [{"is_final": True, "alternatives": [{"transcript": text}]}]}, ensure_ascii=False
            ).encode("utf-8")

        class FakeSock:
            def __init__(self, data: bytes):
                self._data = bytearray(data)

            def settimeout(self, t):
                pass

            def recv(self, n):
                out = bytes(self._data[:7])  # 故意碎片化返回，覆盖跨包拼帧
                del self._data[:7]
                return out

            def sendall(self, b):
                pass

        stream = srv_frame(1, result("第一句。")) + srv_frame(1, result("第二句。")) + srv_frame(8, b"\x03\xe8")
        txt = sogou_mod.SogouEngine._collect_finals(FakeSock(stream), bytearray(), time.monotonic() + 5)
        self.assertEqual(txt, "第一句。第二句。")

        with self.assertRaises(ConnectionError):  # 预热连接在首帧前被回收 -> 可安全重试的 _StaleWSError
            sogou_mod.SogouEngine._collect_finals(FakeSock(b""), bytearray(), time.monotonic() + 5)

    def test_single_engine_split_hedging_cuts_pathological_straggler(self):
        pcm = (self.pcm + b"\x00\x00" * 4000) * 4
        segs = list(audio_mod.iter_vad_segments_streaming(pcm, min_segment_sec=5.0, max_segment_sec=8.0))
        self.assertGreaterEqual(len(segs), 3)
        slow_pcm = segs[1][3]
        calls = []

        def pathological_engine(pcm_bytes: bytes, **kw) -> str:
            calls.append(len(pcm_bytes))
            if pcm_bytes == slow_pcm:
                time.sleep(3.0)  # 服务端对该整段音频的确定性慢解码
                return "整段慢结果。"
            time.sleep(0.05)
            return "片段。"

        with OmniASR() as asr:
            asr.register_engine("pathological", pathological_engine, workers=4)
            got = []
            t0 = time.perf_counter()
            res = asr.transcribe(
                pcm,
                engines=["pathological"],
                min_segment_sec=5.0,
                max_segment_sec=8.0,
                organize=False,
                strict=True,
                on_segment=got.append,
            )
            dt = time.perf_counter() - t0
        self.assertLess(dt, 2.5, f"分裂对冲未生效，总耗时 {dt:.2f}s")
        self.assertEqual(len(got), len(res.segments))
        self.assertEqual(res.segments[1].text, "片段。片段。")
        self.assertGreaterEqual(res.engine_metrics["pathological"]["hedged_wins"], 1)

    def test_llm_fast_failure_retries_immediately_and_falls_back_without_waiting(self):
        org = DoubaoOrganizer(timeout=5.0, hedge_delay=1.0)
        calls = []

        def flaky(body: bytes, timeout: float, fresh: bool = False):
            calls.append(fresh)
            return None if not fresh else "重试成功。"

        with patch.object(org, "_call_once", side_effect=flaky):
            t0 = time.perf_counter()
            self.assertEqual(org.organize_single("测试一下"), "重试成功。")
            self.assertLess(time.perf_counter() - t0, 0.5)  # 主请求失败后立即换新连接重试，不空等 1.0s 对冲时刻
            self.assertEqual(calls, [False, True])

        with patch.object(org, "_call_once", return_value=None):
            t0 = time.perf_counter()
            self.assertEqual(org.organize_single("原文保留"), "原文保留")
            self.assertLess(time.perf_counter() - t0, 0.5)  # 主备均失败：立即降级回原文，不空等至 5s 超时


if __name__ == "__main__":
    unittest.main(verbosity=2)
