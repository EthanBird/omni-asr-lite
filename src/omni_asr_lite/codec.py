# -*- coding: utf-8 -*-
"""
零依赖 WebSocket 帧打包与流式增量文本差分模块 (`omni_asr_lite.codec`)
"""

from __future__ import annotations

import struct
from typing import Union


def pack_ws_frame(payload: Union[bytes, str], opcode: int = 2) -> bytes:
    """打包 RFC 6455 客户端到服务端 WebSocket 帧（采用 4 字节全零 Masking-Key 免除逐字节 XOR 开销）"""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    n = len(raw)
    if n < 126:
        return struct.pack("!BB4s", 0x80 | opcode, 0x80 | n, b"\x00\x00\x00\x00") + raw
    if n < 65536:
        return struct.pack("!BBH4s", 0x80 | opcode, 0x80 | 126, n, b"\x00\x00\x00\x00") + raw
    return struct.pack("!BBQ4s", 0x80 | opcode, 0x80 | 127, n, b"\x00\x00\x00\x00") + raw


_PUNC_SET = frozenset("，。！？、；：,.!?;: ")


def compute_delta(prev: str, curr: str) -> str:
    """计算流式识别相邻两帧文本之间的新增尾部内容 (delta)"""
    if not curr:
        return ""
    if not prev:
        return curr
    if curr.startswith(prev):
        return curr[len(prev) :]
    i, j = 0, 0
    len_p, len_c = len(prev), len(curr)
    while i < len_p and j < len_c:
        if prev[i] == curr[j]:
            i += 1
            j += 1
        elif i + 1 < len_p and j + 1 < len_c and prev[i + 1] == curr[j + 1]:
            i += 2
            j += 2
        elif prev[i] in _PUNC_SET:
            i += 1
        elif curr[j] in _PUNC_SET:
            j += 1
        else:
            break
    return curr[j:] if j > 0 else curr
