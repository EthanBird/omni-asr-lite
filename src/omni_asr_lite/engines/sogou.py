# -*- coding: utf-8 -*-
"""
搜狗输入法 ASR 内置主力引擎 (`SogouEngine`) 与真增量全双工流式连接 (`SogouStreamConnection`)：
1. 支持 14 种语种与汉语方言、热词 `speech_contexts`，采用零依赖 RSA-3072-OAEP-SHA256 + AES-256-CBC。
2. 预握手 TLS 1.3 热连接池带 Liveness 活性检测 (`is_socket_alive`) 与 15s Idle TTL，连接失效自动透明重连。
3. 提供 `open_stream()` 真增量 WebSocket 流式会话：每次 `send_pcm_chunk()` 仅发送新增的 20ms Opus 帧（1.0x 带宽，零重复重传），
   并通过非阻塞 `select` 实时读取服务端 `interim_results` 增量文本。
"""

from __future__ import annotations

import base64
import json
import os
import select
import socket
import ssl
import struct
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..audio import encode_sogou_ws_frames
from ..codec import pack_ws_frame
from ..crypto import aes_256_cbc_encrypt, rsa_oaep_sha256_encrypt
from ..http_pool import is_socket_alive
from .base import BaseASREngine

SOGOU_WS_HOST = "srss.speech.sogou.com"
SOGOU_WS_PATH = "/srss/v1/speech/streaming_recognize"

SOGOU_LANGUAGES: Dict[str, str] = {
    "zh-cmn-Hans-CN": "普通话",
    "zh-cmnxyue-Hans-CN": "粤语",
    "zh-yue-Hans-CN": "粤语(简体)",
    "zh-yue-Hant-HK": "粤语(繁体/香港)",
    "zh-cmn-sichuan": "四川话",
    "zh-cmn-hebei": "河北话",
    "zh-cmn-tianjin": "天津话",
    "zh-cmn-dongbei": "东北话",
    "zh-cmn-nanjing": "南京话",
    "zh-cmn-wuhan": "武汉话",
    "zh-cmn-xian": "西安话",
    "zh-cmn-guizhou": "贵州话",
    "zh-cmn-jinan": "济南话",
    "en-US": "英语",
}

_CANONICAL_LOWER = {k.lower(): k for k in SOGOU_LANGUAGES}
_LANG_ALIAS = {
    "zh": "zh-cmn-Hans-CN",
    "zh-cn": "zh-cmn-Hans-CN",
    "cmn": "zh-cmn-Hans-CN",
    "mandarin": "zh-cmn-Hans-CN",
    "普通话": "zh-cmn-Hans-CN",
    "yue": "zh-cmnxyue-Hans-CN",
    "cantonese": "zh-cmnxyue-Hans-CN",
    "粤语": "zh-cmnxyue-Hans-CN",
    "zh-hk": "zh-yue-Hant-HK",
    "hk": "zh-yue-Hant-HK",
    "粤语繁体": "zh-yue-Hant-HK",
    "sichuan": "zh-cmn-sichuan",
    "sc": "zh-cmn-sichuan",
    "四川话": "zh-cmn-sichuan",
    "hebei": "zh-cmn-hebei",
    "河北话": "zh-cmn-hebei",
    "tianjin": "zh-cmn-tianjin",
    "天津话": "zh-cmn-tianjin",
    "dongbei": "zh-cmn-dongbei",
    "东北话": "zh-cmn-dongbei",
    "nanjing": "zh-cmn-nanjing",
    "南京话": "zh-cmn-nanjing",
    "wuhan": "zh-cmn-wuhan",
    "武汉话": "zh-cmn-wuhan",
    "xian": "zh-cmn-xian",
    "西安话": "zh-cmn-xian",
    "guizhou": "zh-cmn-guizhou",
    "贵州话": "zh-cmn-guizhou",
    "jinan": "zh-cmn-jinan",
    "济南话": "zh-cmn-jinan",
    "en": "en-US",
    "en-us": "en-US",
    "english": "en-US",
    "英语": "en-US",
}


def normalize_sogou_language(language: str) -> str:
    if not language:
        return "zh-cmn-Hans-CN"
    if language in SOGOU_LANGUAGES:
        return language
    norm = language.strip().lower().replace("_", "-")
    if norm in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[norm]
    return _LANG_ALIAS.get(norm, "zh-cmn-Hans-CN")


class SogouStreamConnection:
    """
    搜狗 ASR 真增量全双工 WebSocket 流式会话连接：
    在整个语音段生命周期内保持单条 WebSocket 连接，每次 `send_pcm_chunk()` 仅编码并发送新增的 PCM 字节（零重复发送），
    并通过 `drain_events()` 非阻塞读取服务端实时返回的 `interim_results` 与 `is_final` 消息。
    """

    def __init__(self, sock: ssl.SSLSocket, rbuf: bytearray, cfg_b64: bytes, timeout: float = 15.0):
        self.sock = sock
        self.rbuf = rbuf
        self.timeout = timeout
        self.closed = False
        self._rem_pcm = bytearray()
        self.finals: List[str] = []
        self.latest_partial: str = ""
        # 发送首帧加密配置
        self.sock.sendall(pack_ws_frame(cfg_b64, opcode=1))

    def _read_n(self, k: int) -> bytes:
        while len(self.rbuf) < k:
            c = self.sock.recv(65536)
            if not c:
                raise EOFError()
            self.rbuf.extend(c)
        res = bytes(self.rbuf[:k])
        del self.rbuf[:k]
        return res

    def _consume_one_frame(self) -> Optional[Tuple[str, bool]]:
        b0, b1 = self._read_n(2)
        opcode = b0 & 0x0F
        ln = b1 & 0x7F
        if ln == 126:
            ln = struct.unpack("!H", self._read_n(2))[0]
        elif ln == 127:
            ln = struct.unpack("!Q", self._read_n(8))[0]
        if b1 & 0x80:
            self._read_n(4)
        msg = self._read_n(ln)
        if opcode == 8:
            self.closed = True
            return None
        if opcode == 9:
            self.sock.sendall(pack_ws_frame(msg, opcode=10))
            return ("", False)
        if opcode == 1:
            d = json.loads(msg.decode("utf-8", "ignore"))
            if d.get("error"):
                raise RuntimeError(f"Sogou stream error: {d['error']}")
            got_final = False
            latest_t = ""
            for r in d.get("results", []):
                for alt in r.get("alternatives", []):
                    t = (alt.get("transcript") or "").strip()
                    if r.get("is_final"):
                        got_final = True
                        if t:
                            self.finals.append(t)
                            latest_t = "".join(self.finals)
                    elif t:
                        self.latest_partial = "".join(self.finals) + t
                        latest_t = self.latest_partial
            return (latest_t, got_final)
        return ("", False)

    def send_pcm_chunk(self, pcm_bytes: bytes) -> List[Tuple[str, bool]]:
        """仅将新增的 PCM 字节对齐到 20ms (640B) 边界并增量发送 WebSocket 二进制帧"""
        if self.closed or not pcm_bytes:
            return []
        self._rem_pcm.extend(pcm_bytes)
        aligned_len = (len(self._rem_pcm) // 640) * 640
        if aligned_len > 0:
            to_send = bytes(self._rem_pcm[:aligned_len])
            del self._rem_pcm[:aligned_len]
            ws_frames = encode_sogou_ws_frames(to_send, bitrate=24000)
            if ws_frames:
                self.sock.sendall(ws_frames)
        return self.drain_events(wait_sec=0.02)

    def drain_events(self, wait_sec: float = 0.0) -> List[Tuple[str, bool]]:
        """非阻塞读取当前已到达缓冲区的服务端增量识别结果"""
        events: List[Tuple[str, bool]] = []
        deadline = time.time() + max(0.0, wait_sec)
        while not self.closed:
            has_data = len(self.rbuf) >= 2 or (self.sock.pending() > 0)
            if not has_data:
                rem = max(0.0, deadline - time.time())
                r, _, _ = select.select([self.sock], [], [], rem)
                if not r:
                    break
            try:
                ev = self._consume_one_frame()
            except EOFError:
                self.closed = True
                break
            if ev is None:
                break
            txt, is_fin = ev
            if txt or is_fin:
                events.append((txt, is_fin))
            if time.time() >= deadline and len(self.rbuf) < 2 and self.sock.pending() == 0:
                break
        return events

    def finish(self) -> str:
        """发送残余尾帧与 `{}` 结束标志，等待服务端返回最终 `is_final=True` 文本并关闭连接"""
        try:
            if not self.closed:
                if self._rem_pcm:
                    ws_tail = encode_sogou_ws_frames(bytes(self._rem_pcm), bitrate=24000)
                    self._rem_pcm.clear()
                    if ws_tail:
                        self.sock.sendall(ws_tail)
                self.sock.sendall(pack_ws_frame(b"{}", opcode=1))
                deadline = time.time() + self.timeout
                while not self.closed and time.time() < deadline:
                    try:
                        ev = self._consume_one_frame()
                    except EOFError:
                        break
                    if ev is None:
                        break
                    _, is_fin = ev
                    if is_fin:
                        break
            return "".join(self.finals) if self.finals else self.latest_partial
        finally:
            self.close()

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.close()
        except Exception:
            pass


class SogouEngine(BaseASREngine):
    """搜狗输入法 14 语种/方言高速 ASR 引擎（唯一内置默认引擎）"""

    name = "sogou"

    def __init__(self, timeout: float = 15.0, max_idle_sec: float = 15.0):
        self.timeout = timeout
        self.max_idle_sec = max_idle_sec
        self._ssl_ctx = ssl.create_default_context()
        self._tls_pool: List[Tuple[float, ssl.SSLSocket]] = []
        self._pool_lock = threading.Lock()

    def supports_language(self, language: str) -> bool:
        if not language:
            return True
        norm = language.strip().lower().replace("_", "-")
        return norm in _CANONICAL_LOWER or norm in _LANG_ALIAS or language in SOGOU_LANGUAGES

    def _create_tls_sock(self, timeout: Optional[float] = None) -> ssl.SSLSocket:
        raw_sock = socket.create_connection((SOGOU_WS_HOST, 443), timeout=timeout or self.timeout)
        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self._ssl_ctx.wrap_socket(raw_sock, server_hostname=SOGOU_WS_HOST)

    def _acquire_tls_sock(self, timeout: Optional[float] = None, force_fresh: bool = False) -> ssl.SSLSocket:
        if not force_fresh:
            now = time.monotonic()
            with self._pool_lock:
                while self._tls_pool:
                    ts, s = self._tls_pool.pop()
                    if (now - ts) <= self.max_idle_sec and is_socket_alive(s):
                        try:
                            if timeout:
                                s.settimeout(timeout)
                            return s
                        except Exception:
                            pass
                    try:
                        s.close()
                    except Exception:
                        pass
        return self._create_tls_sock(timeout=timeout)

    def prewarm(self, count: int = 12) -> None:
        def _warm_one():
            try:
                s = self._create_tls_sock(timeout=8.0)
                with self._pool_lock:
                    if len(self._tls_pool) < 24:
                        self._tls_pool.append((time.monotonic(), s))
                        return
                s.close()
            except Exception:
                pass

        for _ in range(max(0, count)):
            threading.Thread(target=_warm_one, daemon=True).start()

    def _build_config_json(
        self,
        language: str = "zh-cmn-Hans-CN",
        hotwords: Optional[List[str]] = None,
        interim_results: bool = False,
    ) -> str:
        lang_code = normalize_sogou_language(language)
        contexts: List[Dict[str, Any]] = []
        if hotwords:
            phrases = [str(w).strip() for w in hotwords if str(w).strip()]
            if phrases:
                contexts.append({"phrases": phrases})
        cfg = {
            "config": {
                "convert_number": False,
                "enable_ambient_sound_event": False,
                "enable_word_time_offsets": False,
                "encoding": "OPUS_WITH_HEADER",
                "first_input_time": 0,
                "ime_install_time": 0,
                "language_code": lang_code,
                "metadata": {
                    "audio_info": {"audio_id": str(uuid.uuid4()), "audio_slice_id": str(uuid.uuid4())},
                    "client_info": {"product_category": "sogou_ime", "product_id": "android_trunk", "product_version": "10.8.0"},
                    "host_device_info": {"device_aid": str(uuid.uuid4()), "device_category": "phone", "device_qid": str(uuid.uuid4()), "device_uuid": str(uuid.uuid4())},
                    "host_os_info": {"os_category": "Android", "os_id": "android", "os_version": "13"},
                    "network_info": {"network_type": "WIFI"},
                    "runtime_info": {},
                    "sdk_info": {"sdk_category": "sogou_ime", "sdk_id": "android_ime", "sdk_version": "46.6.0"},
                    "user_info": {"user_category": "anonymous", "user_id": "anonymous"},
                },
                "model": "",
                "original_audio": False,
                "punctuation_mode": "NORMAL_PUNCTUATION",
                "result_form": "WBEST",
                "speech_contexts": contexts,
                "unit_symbol_type": 0,
                "user_feature": {},
                "custom_info": {},
                "client_itn_switch": True,
                "functions_switch": {},
            },
            "interim_results": bool(interim_results),
            "single_utterance": False,
        }
        return json.dumps(cfg, separators=(",", ":"), ensure_ascii=False)

    def _handshake_ws(
        self,
        cfg_str: str,
        timeout: Optional[float] = None,
    ) -> Tuple[ssl.SSLSocket, bytearray, bytes]:
        aes_key = os.urandom(32)
        iv = os.urandom(16)
        ct = aes_256_cbc_encrypt(aes_key, iv, cfg_str.encode("utf-8"))
        rsa_key = rsa_oaep_sha256_encrypt(aes_key)
        cfg_b64 = base64.b64encode(ct)

        to = timeout or self.timeout
        last_err: Optional[Exception] = None
        # 若从预热池取出的连接因网络闪断握手失败，自动透明退回新建连接重试一次
        for attempt in range(2):
            sock = self._acquire_tls_sock(timeout=to, force_fresh=(attempt > 0))
            if attempt == 0:
                self.prewarm(1)
            try:
                ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
                req = (
                    f"GET {SOGOU_WS_PATH} HTTP/1.1\r\n"
                    f"Host: {SOGOU_WS_HOST}\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {ws_key}\r\n"
                    "Sec-WebSocket-Version: 13\r\n"
                    "User-Agent: okhttp/4.9.3\r\n"
                    "X-Srss-Cipher-Key-Type: 1\r\n"
                    f"X-Srss-Cipher-Key-Sec: {base64.b64encode(rsa_key).decode('ascii')}\r\n"
                    f"X-Srss-Cipher-Key-Vec: {base64.b64encode(iv).decode('ascii')}\r\n\r\n"
                ).encode("ascii")
                sock.sendall(req)

                hbuf = bytearray()
                while b"\r\n\r\n" not in hbuf:
                    ch = sock.recv(4096)
                    if not ch:
                        raise ConnectionError("Sogou WS 握手提前关闭")
                    hbuf.extend(ch)
                sep = hbuf.index(b"\r\n\r\n")
                if "101" not in hbuf[:sep].splitlines()[0].decode("latin1", "ignore"):
                    raise ConnectionError("Sogou WS 握手拒绝")
                rbuf = bytearray(hbuf[sep + 4 :])
                return sock, rbuf, cfg_b64
            except Exception as e:
                last_err = e
                try:
                    sock.close()
                except Exception:
                    pass
        assert last_err is not None
        raise last_err

    def open_stream(
        self,
        *,
        language: str = "zh",
        hotwords: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> SogouStreamConnection:
        """打开一条真增量全双工 WebSocket 识别流（开启 `interim_results=True`）"""
        cfg_str = self._build_config_json(language=language, hotwords=hotwords, interim_results=True)
        sock, rbuf, cfg_b64 = self._handshake_ws(cfg_str, timeout=timeout)
        return SogouStreamConnection(sock, rbuf, cfg_b64, timeout=timeout or self.timeout)

    def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str = "zh",
        hotwords: Optional[List[str]] = None,
        context: str = "",
        timeout: Optional[float] = None,
    ) -> str:
        _ = context
        ws_audio_frames = encode_sogou_ws_frames(pcm_bytes, bitrate=24000)
        if not ws_audio_frames:
            return ""
        cfg_str = self._build_config_json(language=language, hotwords=hotwords, interim_results=False)
        to = timeout or self.timeout
        sock, rbuf, cfg_b64 = self._handshake_ws(cfg_str, timeout=to)
        try:
            all_out = (
                pack_ws_frame(cfg_b64, opcode=1)
                + ws_audio_frames
                + pack_ws_frame(b"{}", opcode=1)
            )
            sock.sendall(all_out)

            def read_n(k: int) -> bytes:
                while len(rbuf) < k:
                    c = sock.recv(65536)
                    if not c:
                        raise EOFError()
                    rbuf.extend(c)
                res = bytes(rbuf[:k])
                del rbuf[:k]
                return res

            finals: List[str] = []
            deadline = time.time() + to
            while time.time() < deadline:
                try:
                    b0, b1 = read_n(2)
                except EOFError:
                    break
                opcode = b0 & 0x0F
                ln = b1 & 0x7F
                if ln == 126:
                    ln = struct.unpack("!H", read_n(2))[0]
                elif ln == 127:
                    ln = struct.unpack("!Q", read_n(8))[0]
                if b1 & 0x80:
                    read_n(4)
                msg = read_n(ln)
                if opcode == 8:
                    break
                if opcode == 9:
                    sock.sendall(pack_ws_frame(msg, opcode=10))
                    continue
                if opcode == 1:
                    d = json.loads(msg.decode("utf-8", "ignore"))
                    if d.get("error"):
                        raise RuntimeError(f"Sogou error: {d['error']}")
                    got_final = False
                    for r in d.get("results", []):
                        for alt in r.get("alternatives", []):
                            t = (alt.get("transcript") or "").strip()
                            if r.get("is_final"):
                                got_final = True
                                if t:
                                    finals.append(t)
                    if got_final:
                        break
            return "".join(finals)
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._pool_lock:
            socks = [s for _, s in self._tls_pool]
            self._tls_pool.clear()
        for s in socks:
            try:
                s.close()
            except Exception:
                pass
