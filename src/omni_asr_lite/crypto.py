# -*- coding: utf-8 -*-
"""
零依赖密码学模块 (`omni_asr_lite.crypto`)：
1. AES-256-CBC + PKCS7：
   - Windows 优先使用系统原生 CNG `bcrypt.dll` 硬件加速
   - 兜底使用纯 Python 查表法 AES-256（100% 跨平台零依赖）
2. RSA-3072 OAEP-SHA256 公钥加密（基于标准库 `hashlib.sha256` + `pow(m, e, n)`）
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import sys
import threading
from typing import Dict, Tuple


class _WinBCryptAES:
    def __init__(self):
        self._bcrypt = ctypes.windll.bcrypt
        self._hAlg = ctypes.c_void_p()
        status = self._bcrypt.BCryptOpenAlgorithmProvider(
            ctypes.byref(self._hAlg), ctypes.c_wchar_p("AES"), None, 0
        )
        if status != 0:
            raise OSError(f"BCryptOpenAlgorithmProvider 失败: {status:#x}")

        mode = "ChainingModeECB".encode("utf-16le") + b"\x00\x00"
        status = self._bcrypt.BCryptSetProperty(
            self._hAlg, ctypes.c_wchar_p("ChainingMode"), mode, len(mode), 0
        )
        if status != 0:
            raise OSError(f"BCryptSetProperty 失败: {status:#x}")

        obj_len = ctypes.c_ulong(0)
        res_len = ctypes.c_ulong(0)
        self._bcrypt.BCryptGetProperty(
            self._hAlg,
            ctypes.c_wchar_p("ObjectLength"),
            ctypes.byref(obj_len),
            4,
            ctypes.byref(res_len),
            0,
        )
        self._obj_len = obj_len.value or 1024
        self._lock = threading.Lock()
        self._cache: Dict[bytes, Tuple[ctypes.c_void_p, ctypes.Array]] = {}

    def _get_key_handle(self, key: bytes) -> ctypes.c_void_p:
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached[0]
            if len(self._cache) >= 32:
                for h, _ in self._cache.values():
                    self._bcrypt.BCryptDestroyKey(h)
                self._cache.clear()
            key_obj = ctypes.create_string_buffer(self._obj_len)
            blob = struct.pack("<III", 0x4D42444B, 1, len(key)) + key
            hKey = ctypes.c_void_p()
            status = self._bcrypt.BCryptImportKey(
                self._hAlg, None, ctypes.c_wchar_p("KeyDataBlob"),
                ctypes.byref(hKey), key_obj, self._obj_len, blob, len(blob), 0,
            )
            if status != 0:
                raise OSError(f"BCryptImportKey 失败: {status:#x}")
            self._cache[key] = (hKey, key_obj)
            return hKey

    def encrypt_ecb(self, key: bytes, padded: bytes) -> bytes:
        hKey = self._get_key_handle(key)
        n = len(padded)
        out = ctypes.create_string_buffer(n)
        cb_res = ctypes.c_ulong(0)
        status = self._bcrypt.BCryptEncrypt(
            hKey, padded, n, None, None, 0, out, n, ctypes.byref(cb_res), 0
        )
        if status != 0:
            raise OSError(f"BCryptEncrypt 失败: {status:#x}")
        return out.raw[: cb_res.value]


_SBOX = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)
_RCON = (0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)


def _xtime(a: int) -> int:
    return ((a << 1) ^ 0x1B) & 0xFF if (a & 0x80) else (a << 1)


class _PurePythonAES256:
    @staticmethod
    def _expand_key(key: bytes):
        w = [list(key[i : i + 4]) for i in range(0, 32, 4)]
        for i in range(8, 60):
            temp = list(w[i - 1])
            if i % 8 == 0:
                temp = [_SBOX[temp[1]] ^ _RCON[i // 8], _SBOX[temp[2]], _SBOX[temp[3]], _SBOX[temp[0]]]
            elif i % 8 == 4:
                temp = [_SBOX[b] for b in temp]
            w.append([w[i - 8][j] ^ temp[j] for j in range(4)])
        return [sum((w[r * 4 + c] for c in range(4)), []) for r in range(15)]

    @classmethod
    def encrypt_ecb(cls, key: bytes, padded: bytes) -> bytes:
        rkeys = cls._expand_key(key)
        out = bytearray()
        for off in range(0, len(padded), 16):
            s = [padded[off + j] ^ rkeys[0][j] for j in range(16)]
            for r in range(1, 14):
                s = [_SBOX[b] for b in s]
                s = [
                    s[0], s[5], s[10], s[15],
                    s[4], s[9], s[14], s[3],
                    s[8], s[13], s[2], s[7],
                    s[12], s[1], s[6], s[11],
                ]
                for c in range(0, 16, 4):
                    a0, a1, a2, a3 = s[c], s[c + 1], s[c + 2], s[c + 3]
                    t = a0 ^ a1 ^ a2 ^ a3
                    s[c] ^= t ^ _xtime(a0 ^ a1)
                    s[c + 1] ^= t ^ _xtime(a1 ^ a2)
                    s[c + 2] ^= t ^ _xtime(a2 ^ a3)
                    s[c + 3] ^= t ^ _xtime(a3 ^ a0)
                rk = rkeys[r]
                s = [s[j] ^ rk[j] for j in range(16)]
            s = [_SBOX[b] for b in s]
            s = [
                s[0], s[5], s[10], s[15],
                s[4], s[9], s[14], s[3],
                s[8], s[13], s[2], s[7],
                s[12], s[1], s[6], s[11],
            ]
            rk = rkeys[14]
            out.extend(s[j] ^ rk[j] for j in range(16))
        return bytes(out)


def _init_aes_backend():
    if sys.platform == "win32":
        try:
            return _WinBCryptAES()
        except Exception:
            pass
    return _PurePythonAES256()


_AES_BACKEND = _init_aes_backend()


def aes_256_cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """AES-256-CBC + PKCS7 填充加密"""
    pad_len = 16 - (len(plaintext) % 16)
    padded = plaintext + bytes([pad_len]) * pad_len
    out = bytearray()
    prev = iv
    enc_fn = _AES_BACKEND.encrypt_ecb
    for off in range(0, len(padded), 16):
        xored = bytes(a ^ b for a, b in zip(padded[off : off + 16], prev))
        enc_blk = enc_fn(key, xored)
        out.extend(enc_blk)
        prev = enc_blk
    return bytes(out)


def _parse_sogou_der_rsa_public_key() -> Tuple[int, int, int]:
    raw_list = [
        48, -126, 1, -94, 48, 13, 6, 9, 42, -122, 72, -122, -9, 13, 1, 1, 1, 5, 0, 3,
        -126, 1, -113, 0, 48, -126, 1, -118, 2, -126, 1, -127, 0, -45, -115, -80, 123,
        75, 105, -43, 7, -3, -96, -98, 71, 61, 59, 67, 109, -21, -17, -105, -49, 55,
        -85, -9, -73, 21, 117, -16, -99, 6, -60, -19, -66, 41, -24, -22, -83, 86, 52,
        49, -90, -80, -82, -80, -46, -126, -120, -13, -36, -73, 53, -108, -52, -23,
        -117, 10, 122, 51, 90, -97, -5, -113, 123, 17, 57, -44, -61, 60, -82, -39, -76,
        73, 95, -23, -14, -53, -6, 93, -101, 109, -47, -30, -115, -12, -85, -47, -97,
        124, -56, -68, 16, 58, -23, -114, 56, -11, -83, 115, 20, -9, -31, 43, 43, -85,
        -111, -73, 74, -90, -80, -13, -12, 66, -11, -103, 110, -96, -29, 106, -94, -20,
        -60, -105, 102, -82, -80, -6, 54, -43, 111, -78, 114, -39, -120, -51, 15, 40,
        -76, -62, -114, -111, 59, -89, 53, 40, 62, 127, -106, -52, -18, -104, 97, -79,
        -118, 13, -53, -119, -28, 10, 34, 18, -71, 120, -27, 86, -111, 39, -67, -65,
        85, -46, -34, 34, -121, 107, 0, 2, -6, -126, -104, 104, -90, -32, -12, 14, 12,
        -102, -125, 15, 62, -68, 35, 61, 26, -33, 105, 12, 80, -124, 117, -55, -45, 42,
        -114, -127, -24, -111, 23, 78, -109, -11, 87, -126, 49, 59, 58, 98, 108, 85,
        87, -69, 22, 126, 30, -51, -40, 59, -123, -121, -25, 41, 80, -113, 39, 27, -13,
        -121, 71, 32, 117, -80, -116, 81, 68, -105, 56, -1, 31, -96, -87, -55, 16, 74,
        6, 33, 7, 21, -6, 86, 32, 42, -93, 84, 72, -110, 81, -83, -4, -58, -33, 120,
        -71, -15, -92, 112, 39, 111, 84, 57, 81, -18, -124, 98, 127, 74, -95, 87, -88,
        -39, -109, -67, -124, 17, -25, -10, 60, -110, -13, 2, -95, 55, 113, 64, 110,
        -100, -27, -75, -78, -10, 52, -34, -78, -73, 63, 58, 88, -98, -52, 67, 18, -12,
        105, 94, -91, -24, -115, 118, 81, -31, -35, -54, 46, 15, 41, -10, -49, 38, 48,
        19, -90, -127, -90, -86, 123, -109, 123, -107, 66, -65, -5, 62, 32, 9, -23,
        -112, -63, 66, 76, -43, 95, 99, -58, -54, -95, 94, 2, -125, -59, 19, -116, 96,
        -24, -78, -45, 96, 126, 4, -55, -42, -23, 101, 114, 3, -125, -116, -13, 2, 3,
        1, 0, 1,
    ]
    der = bytes((b + 256) % 256 for b in raw_list)
    idx = der.index(b"\x02\x82\x01\x81\x00")
    n_bytes = der[idx + 5 : idx + 5 + 384]
    return int.from_bytes(n_bytes, "big"), 65537, len(n_bytes)


RSA_N, RSA_E, RSA_K = _parse_sogou_der_rsa_public_key()
_EMPTY_SHA256 = hashlib.sha256(b"").digest()


def _mgf1_sha256(seed: bytes, length: int) -> bytes:
    out = bytearray()
    for counter in range((length + 31) // 32):
        out.extend(hashlib.sha256(seed + struct.pack(">I", counter)).digest())
    return bytes(out[:length])


def rsa_oaep_sha256_encrypt(msg: bytes) -> bytes:
    ps_len = RSA_K - len(msg) - 66
    db = _EMPTY_SHA256 + (b"\x00" * ps_len) + b"\x01" + msg
    seed = os.urandom(32)
    masked_db = bytes(a ^ b for a, b in zip(db, _mgf1_sha256(seed, RSA_K - 33)))
    masked_seed = bytes(a ^ b for a, b in zip(seed, _mgf1_sha256(masked_db, 32)))
    m_int = int.from_bytes(b"\x00" + masked_seed + masked_db, "big")
    return pow(m_int, RSA_E, RSA_N).to_bytes(RSA_K, "big")
