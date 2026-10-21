# -*- coding: utf-8 -*-
"""
通用零依赖 HTTPS Keep-Alive 连接池 (`HTTPSConnectionPool`)：
供豆包官方 LLM「智能整理」接口与自定义扩展引擎复用。
"""

from __future__ import annotations

import http.client
import socket
import ssl
import threading
from typing import Dict, List, Optional, Tuple


class HTTPSConnectionPool:
    """线程安全的 Keep-Alive HTTPS 连接池"""

    def __init__(self, host: str, timeout: float = 20.0, max_pool_size: int = 20):
        self.host = host
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self._ssl_ctx = ssl.create_default_context()
        self._pool: List[http.client.HTTPSConnection] = []
        self._lock = threading.Lock()

    def _create_conn(self, timeout: Optional[float] = None) -> http.client.HTTPSConnection:
        conn = http.client.HTTPSConnection(
            self.host, timeout=timeout or self.timeout, context=self._ssl_ctx
        )
        conn.connect()
        if conn.sock is not None:
            try:
                conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
        return conn

    def _acquire(self, timeout: Optional[float] = None) -> http.client.HTTPSConnection:
        with self._lock:
            if self._pool:
                conn = self._pool.pop()
                if timeout and conn.sock is not None:
                    try:
                        conn.sock.settimeout(timeout)
                    except Exception:
                        pass
                return conn
        return self._create_conn(timeout=timeout)

    def _release(self, conn: http.client.HTTPSConnection) -> None:
        if conn.sock is None:
            return
        with self._lock:
            if len(self._pool) < self.max_pool_size:
                self._pool.append(conn)
                return
        try:
            conn.close()
        except Exception:
            pass

    def post(
        self,
        path: str,
        body: bytes,
        headers: Dict[str, str],
        timeout: Optional[float] = None,
    ) -> Tuple[int, bytes]:
        req_headers = dict(headers)
        req_headers["Connection"] = "keep-alive"
        req_headers["Content-Length"] = str(len(body))

        for attempt in range(2):
            conn = self._acquire(timeout=timeout) if attempt == 0 else self._create_conn(timeout=timeout)
            try:
                conn.request("POST", path, body=body, headers=req_headers)
                resp = conn.getresponse()
                data = resp.read()
                status = resp.status
                self._release(conn)
                return status, data
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                if attempt == 1:
                    return 0, b""
        return 0, b""

    def close_all(self) -> None:
        with self._lock:
            conns = list(self._pool)
            self._pool.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
