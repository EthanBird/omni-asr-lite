# -*- coding: utf-8 -*-
"""
搜狗输入法 ASR 内置主力引擎 (`SogouEngine`) 与真增量全双工流式连接 (`SogouStreamConnection`)：
1. 支持 14 种语种与汉语方言、热词 `speech_contexts`，采用零依赖 RSA-3072-OAEP-SHA256 + AES-256-CBC。
2. **预升级 WebSocket 热连接池 (Ready-WS Pool)**：后台线程提前完成「TCP + TLS 1.3 + RSA 密钥协商 + WS Upgrade」全部握手，
   取用时仅需 AES 加密配置帧（~5µs）即可直接发送音频，彻底移出每段 ~90ms 的 Upgrade RTT；
   带 5s Idle TTL（实测服务端约 9~10s 回收空闲已升级连接）+ Liveness 活性检测，陈旧连接自动透明回退冷握手重试。
3. **边编码边发送**：Opus 编码按 1s 分块与网络上行 / 服务端解码完全重叠；冷握手路径则在等待 `101` 响应期间完成编码。
4. **多句定稿完整收集**：服务端会按其内部 VAD 对单段音频输出多条 `is_final` 结果并随后发送 Close 帧，
   本引擎读取至 Close 为止并拼接全部定稿句，杜绝「只取第一句」造成的漏字。
5. 提供 `open_stream()` 真增量 WebSocket 流式会话：每次 `send_pcm_chunk()` 仅发送新增的 20ms Opus 帧（1.0x 带宽，零重复重传），
   独占编码器保持跨 chunk 连续状态，并通过非阻塞 `select` 实时读取服务端 `interim_results` 增量文本。
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

from ..audio import SogouWSFrameEncoder, encode_sogou_ws_frames, iter_sogou_ws_frames
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

_WS_END_FRAME = pack_ws_frame(b"{}", opcode=1)
_UNPACK_U16 = struct.Struct("!H").unpack
_UNPACK_U64 = struct.Struct("!Q").unpack
# 收到首条 is_final 后若服务端迟迟不发 Close，最多再等待该时长收集后续定稿句（实测 Close 通常在最后一句后 0~2ms 内到达）
_FINAL_LINGER_SEC = 1.5


def normalize_sogou_language(language: str) -> str:
    if not language:
        return "zh-cmn-Hans-CN"
    if language in SOGOU_LANGUAGES:
        return language
    norm = language.strip().lower().replace("_", "-")
    if norm in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[norm]
    return _LANG_ALIAS.get(norm, "zh-cmn-Hans-CN")


class _StaleWSError(ConnectionError):
    """预热连接在收到任何 WebSocket 帧之前即被对端关闭（空闲回收），可安全换新连接重试"""


class _ReadyWS:
    """已完成 TLS + WS Upgrade 握手、等待发送加密配置帧的热连接"""

    __slots__ = ("ts", "sock", "rbuf", "aes_key", "iv")

    def __init__(self, ts: float, sock: ssl.SSLSocket, rbuf: bytearray, aes_key: bytes, iv: bytes):
        self.ts = ts
        self.sock = sock
        self.rbuf = rbuf
        self.aes_key = aes_key
        self.iv = iv

    def config_frame(self, cfg_plain: bytes) -> bytes:
        ct = aes_256_cbc_encrypt(self.aes_key, self.iv, cfg_plain)
        return pack_ws_frame(base64.b64encode(ct), opcode=1)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def _recv_ws_frame(sock: ssl.SSLSocket, rbuf: bytearray) -> Tuple[int, bytes]:
    """从 `rbuf` + socket 中读取一个完整 WebSocket 帧，返回 `(opcode, payload)`；对端关闭时抛出 EOFError"""

    def need(k: int) -> None:
        while len(rbuf) < k:
            c = sock.recv(65536)
            if not c:
                raise EOFError()
            rbuf.extend(c)

    need(2)
    b0, b1 = rbuf[0], rbuf[1]
    ln = b1 & 0x7F
    pos = 2
    if ln == 126:
        need(4)
        ln = _UNPACK_U16(bytes(rbuf[2:4]))[0]
        pos = 4
    elif ln == 127:
        need(10)
        ln = _UNPACK_U64(bytes(rbuf[2:10]))[0]
        pos = 10
    if b1 & 0x80:
        pos += 4
    need(pos + ln)
    msg = bytes(rbuf[pos : pos + ln])
    del rbuf[: pos + ln]
    return b0 & 0x0F, msg


def _parse_results(msg: bytes) -> Tuple[List[str], str, bool]:
    """解析服务端文本帧，返回 `(本帧定稿句列表, 本帧最新中间结果, 是否含 is_final)`"""
    d = json.loads(msg.decode("utf-8", "ignore"))
    if d.get("error"):
        raise RuntimeError(f"Sogou error: {d['error']}")
    finals: List[str] = []
    partial = ""
    got_final = False
    for r in d.get("results", []) or ():
        is_final = bool(r.get("is_final"))
        for alt in r.get("alternatives", []) or ():
            t = (alt.get("transcript") or "").strip()
            if is_final:
                got_final = True
                if t:
                    finals.append(t)
            elif t:
                partial = t
    return finals, partial, got_final


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
        self._encoder: Optional[SogouWSFrameEncoder] = None
        # 发送首帧加密配置
        self.sock.sendall(pack_ws_frame(cfg_b64, opcode=1))

    def _encode(self, pcm: bytes) -> bytes:
        if self._encoder is None:
            self._encoder = SogouWSFrameEncoder(bitrate=24000)
        return self._encoder.encode(pcm)

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
        opcode, msg = _recv_ws_frame(self.sock, self.rbuf)
        if opcode == 8:
            self.closed = True
            return None
        if opcode == 9:
            self.sock.sendall(pack_ws_frame(msg, opcode=10))
            return ("", False)
        if opcode == 1:
            finals, partial, got_final = _parse_results(msg)
            latest_t = ""
            if finals:
                self.finals.extend(finals)
                latest_t = "".join(self.finals)
            if partial:
                self.latest_partial = "".join(self.finals) + partial
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
            ws_frames = self._encode(to_send)
            if ws_frames:
                self.sock.sendall(ws_frames)
        return self.drain_events(wait_sec=0.02)

    def drain_events(self, wait_sec: float = 0.0) -> List[Tuple[str, bool]]:
        """非阻塞读取当前已到达缓冲区的服务端增量识别结果"""
        events: List[Tuple[str, bool]] = []
        deadline = time.monotonic() + max(0.0, wait_sec)
        while not self.closed:
            has_data = len(self.rbuf) >= 2 or (self.sock.pending() > 0)
            if not has_data:
                rem = max(0.0, deadline - time.monotonic())
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
            if time.monotonic() >= deadline and len(self.rbuf) < 2 and self.sock.pending() == 0:
                break
        return events

    def finish(self) -> str:
        """发送残余尾帧与 `{}` 结束标志，读取服务端全部 `is_final` 定稿句直至 Close 帧并关闭连接"""
        try:
            if not self.closed:
                if self._rem_pcm:
                    ws_tail = self._encode(bytes(self._rem_pcm))
                    self._rem_pcm.clear()
                    if ws_tail:
                        self.sock.sendall(ws_tail)
                self.sock.sendall(_WS_END_FRAME)
                deadline = time.monotonic() + self.timeout
                linger_until = 0.0
                while not self.closed:
                    now = time.monotonic()
                    limit = min(deadline, linger_until) if linger_until else deadline
                    if now >= limit:
                        break
                    self.sock.settimeout(max(0.01, limit - now))
                    try:
                        ev = self._consume_one_frame()
                    except (EOFError, socket.timeout):
                        break
                    if ev is None:
                        break
                    _, is_fin = ev
                    if is_fin and not linger_until:
                        linger_until = time.monotonic() + _FINAL_LINGER_SEC
            return "".join(self.finals) if self.finals else self.latest_partial
        finally:
            self.close()

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.close()
        except Exception:
            pass
        enc, self._encoder = self._encoder, None
        if enc is not None:
            enc.close()


class SogouEngine(BaseASREngine):
    """搜狗输入法 14 语种/方言高速 ASR 引擎（唯一内置默认引擎）"""

    name = "sogou"

    def __init__(
        self,
        timeout: float = 15.0,
        max_idle_sec: float = 5.0,
        max_ready: int = 32,
    ):
        self.timeout = timeout
        # 预升级 WS 连接的空闲 TTL：实测服务端会在 ~9-10s 后回收空闲的已升级连接，这里取 5s 留足安全余量
        self.max_idle_sec = max_idle_sec
        self.max_ready = max(1, int(max_ready))
        self._ssl_ctx = ssl.create_default_context()
        self._ws_pool: List[_ReadyWS] = []
        self._pool_lock = threading.Lock()
        self._warming = 0
        self._target_ready = 0
        self._closed = False

    def supports_language(self, language: str) -> bool:
        if not language:
            return True
        norm = language.strip().lower().replace("_", "-")
        return norm in _CANONICAL_LOWER or norm in _LANG_ALIAS or language in SOGOU_LANGUAGES

    # -------------------- 连接建立与预升级热连接池 --------------------
    def _create_tls_sock(self, timeout: Optional[float] = None) -> ssl.SSLSocket:
        raw_sock = socket.create_connection((SOGOU_WS_HOST, 443), timeout=timeout or self.timeout)
        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            return self._ssl_ctx.wrap_socket(raw_sock, server_hostname=SOGOU_WS_HOST)
        except BaseException:
            raw_sock.close()
            raise

    @staticmethod
    def _send_upgrade(sock: ssl.SSLSocket, aes_key: bytes, iv: bytes) -> None:
        rsa_key = rsa_oaep_sha256_encrypt(aes_key)
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

    @staticmethod
    def _recv_upgrade(sock: ssl.SSLSocket) -> bytearray:
        hbuf = bytearray()
        while b"\r\n\r\n" not in hbuf:
            ch = sock.recv(4096)
            if not ch:
                raise ConnectionError("Sogou WS 握手提前关闭")
            hbuf.extend(ch)
        sep = hbuf.index(b"\r\n\r\n")
        if b"101" not in bytes(hbuf[:sep]).split(b"\r\n", 1)[0]:
            raise ConnectionError("Sogou WS 握手拒绝")
        return bytearray(hbuf[sep + 4 :])

    def _open_ready_ws(self, timeout: Optional[float] = None) -> _ReadyWS:
        aes_key = os.urandom(32)
        iv = os.urandom(16)
        sock = self._create_tls_sock(timeout=timeout)
        try:
            self._send_upgrade(sock, aes_key, iv)
            rbuf = self._recv_upgrade(sock)
        except BaseException:
            try:
                sock.close()
            except Exception:
                pass
            raise
        return _ReadyWS(time.monotonic(), sock, rbuf, aes_key, iv)

    def _pop_ready_ws(self) -> Optional[_ReadyWS]:
        now = time.monotonic()
        stale: List[_ReadyWS] = []
        picked: Optional[_ReadyWS] = None
        with self._pool_lock:
            while self._ws_pool:
                e = self._ws_pool.pop()  # LIFO：优先取最新鲜的连接
                if (now - e.ts) <= self.max_idle_sec and not e.rbuf and is_socket_alive(e.sock):
                    picked = e
                    break
                stale.append(e)
            if picked is None:
                # 栈顶都已过期，则更早入池的连接必然也已过期
                stale.extend(self._ws_pool)
                self._ws_pool.clear()
        for e in stale:
            e.close()
        return picked

    def _fill_to_target_locked(self, target: int) -> None:
        """（需持有 `_pool_lock`）剔除过期连接后，按 `target - 池中可用数 - 正在预热数` 的差额启动后台预热线程"""
        now = time.monotonic()
        if self._ws_pool and (now - self._ws_pool[0].ts) > self.max_idle_sec:
            keep: List[_ReadyWS] = []
            for e in self._ws_pool:
                if (now - e.ts) <= self.max_idle_sec:
                    keep.append(e)
                else:
                    e.close()  # 仅关闭本地 fd，不做 TLS close_notify 往返，不会阻塞
            self._ws_pool[:] = keep
        for _ in range(max(0, target - len(self._ws_pool) - self._warming)):
            self._warming += 1
            threading.Thread(target=self._warm_one, name="omni-lite-sogou-warm", daemon=True).start()

    def _warm_one(self) -> None:
        entry: Optional[_ReadyWS] = None
        try:
            entry = self._open_ready_ws(timeout=8.0)
        except Exception:
            entry = None
        finally:
            with self._pool_lock:
                self._warming -= 1
                if entry is not None and not self._closed and len(self._ws_pool) < self.max_ready:
                    self._ws_pool.append(entry)
                    entry = None
        if entry is not None:
            entry.close()

    def _replenish(self, min_target: int = 1) -> None:
        with self._pool_lock:
            if self._closed:
                return
            self._fill_to_target_locked(min(self.max_ready, max(self._target_ready, min_target)))

    def prewarm(self, count: int = 12) -> None:
        """后台预建 `count` 条已完成 TLS + WS Upgrade 的热连接（按差额补齐，重复调用不会超建）"""
        with self._pool_lock:
            self._closed = False  # 显式预热视为重新启用（允许 close() 之后复用引擎）
            self._target_ready = min(self.max_ready, max(0, int(count)))
            self._fill_to_target_locked(self._target_ready)

    def warm_one_ahead(self) -> None:
        """不改变池目标规模，仅在池中无可用连接时补建 1 条热连接（供流式会话在句段切换前预取下一条连接）"""
        with self._pool_lock:
            if not self._closed:
                self._fill_to_target_locked(1)

    def _acquire_ready_ws(self, timeout: Optional[float] = None, replenish: bool = True) -> Tuple[_ReadyWS, bool]:
        """取一条可用热连接；池空时同步冷握手。返回 `(连接, 是否来自预热池)`"""
        e = self._pop_ready_ws()
        if replenish:
            self._replenish()
        if e is not None:
            if timeout:
                e.sock.settimeout(timeout)
            return e, True
        return self._open_ready_ws(timeout=timeout), False

    def _handshake_ws(
        self,
        cfg_str: str,
        timeout: Optional[float] = None,
    ) -> Tuple[ssl.SSLSocket, bytearray, bytes]:
        """兼容接口：返回已完成 Upgrade 的 `(sock, rbuf, 加密配置 base64)`"""
        to = timeout or self.timeout
        e, _ = self._acquire_ready_ws(timeout=to)
        cfg_b64 = base64.b64encode(aes_256_cbc_encrypt(e.aes_key, e.iv, cfg_str.encode("utf-8")))
        return e.sock, e.rbuf, cfg_b64

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

    def open_stream(
        self,
        *,
        language: str = "zh",
        hotwords: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> SogouStreamConnection:
        """打开一条真增量全双工 WebSocket 识别流（开启 `interim_results=True`）"""
        cfg_str = self._build_config_json(language=language, hotwords=hotwords, interim_results=True)
        to = timeout or self.timeout
        for attempt in range(2):
            # 流式会话的句段长达 5~14s，常规补齐的热连接在下次开流时早已过期；改由会话在句段切换前显式 warm_one_ahead()
            e, from_pool = self._acquire_ready_ws(timeout=to, replenish=False)
            cfg_b64 = base64.b64encode(aes_256_cbc_encrypt(e.aes_key, e.iv, cfg_str.encode("utf-8")))
            try:
                return SogouStreamConnection(e.sock, e.rbuf, cfg_b64, timeout=to)
            except (OSError, ssl.SSLError):
                e.close()
                if not from_pool or attempt:
                    raise
        raise ConnectionError("Sogou 流式连接建立失败")  # pragma: no cover

    # -------------------- 单段识别 --------------------
    @staticmethod
    def _collect_finals(sock: ssl.SSLSocket, rbuf: bytearray, deadline: float) -> str:
        """读取服务端全部 `is_final` 定稿句直至 Close 帧 / EOF / 超时；首帧前即 EOF 时抛出 `_StaleWSError`"""
        finals: List[str] = []
        got_any = False
        linger_until = 0.0
        while True:
            now = time.monotonic()
            limit = min(deadline, linger_until) if linger_until else deadline
            if now >= limit:
                break
            sock.settimeout(max(0.01, limit - now))
            try:
                opcode, msg = _recv_ws_frame(sock, rbuf)
            except EOFError:
                if not got_any:
                    raise _StaleWSError("Sogou WS 连接在返回结果前被关闭") from None
                break
            except socket.timeout:
                if linger_until:
                    break
                raise
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
                if not got_any:
                    raise _StaleWSError(str(exc)) from None
                raise
            got_any = True
            if opcode == 8:
                break
            if opcode == 9:
                sock.sendall(pack_ws_frame(msg, opcode=10))
                continue
            if opcode == 1:
                seg_finals, _, got_final = _parse_results(msg)
                finals.extend(seg_finals)
                if got_final and not linger_until:
                    linger_until = time.monotonic() + _FINAL_LINGER_SEC
        return "".join(finals)

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
        if not pcm_bytes:
            return ""
        to = timeout or self.timeout
        deadline = time.monotonic() + to
        cfg_plain = self._build_config_json(language=language, hotwords=hotwords, interim_results=False).encode("utf-8")
        encoded: Optional[bytes] = None

        # ---- 热路径：预升级连接 + 边编码边发送 ----
        e = self._pop_ready_ws()
        self._replenish()
        if e is not None:
            parts: List[bytes] = []
            try:
                sock = e.sock
                sock.settimeout(to)
                head = e.config_frame(cfg_plain)
                for blk in iter_sogou_ws_frames(pcm_bytes, bitrate=24000, chunk_frames=50):
                    parts.append(blk)
                    if head:
                        sock.sendall(head + blk)
                        head = b""
                    else:
                        sock.sendall(blk)
                sock.sendall(_WS_END_FRAME)
                return self._collect_finals(sock, e.rbuf, deadline)
            except (
                _StaleWSError,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                ssl.SSLEOFError,
                ssl.SSLZeroReturnError,
            ):
                # 预热连接已被服务端空闲回收：透明回退到冷握手重试一次（若已完整编码则直接复用编码结果）
                if len(parts) == -(-len(pcm_bytes) // 32000):
                    encoded = b"".join(parts)
            finally:
                e.close()
            if time.monotonic() >= deadline:
                raise TimeoutError("Sogou ASR 超时")

        # ---- 冷路径：发送 Upgrade 请求后，在等待 101 响应的 RTT 内完成 Opus 编码 ----
        aes_key = os.urandom(32)
        iv = os.urandom(16)
        sock = self._create_tls_sock(timeout=max(1.0, deadline - time.monotonic()))
        try:
            self._send_upgrade(sock, aes_key, iv)
            if encoded is None:
                encoded = encode_sogou_ws_frames(pcm_bytes, bitrate=24000)
            rbuf = self._recv_upgrade(sock)
            cfg_frame = pack_ws_frame(base64.b64encode(aes_256_cbc_encrypt(aes_key, iv, cfg_plain)), opcode=1)
            sock.sendall(cfg_frame + encoded + _WS_END_FRAME)
            try:
                return self._collect_finals(sock, rbuf, deadline)
            except _StaleWSError as exc:
                raise ConnectionError(str(exc)) from None
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            entries = list(self._ws_pool)
            self._ws_pool.clear()
        for e in entries:
            e.close()
