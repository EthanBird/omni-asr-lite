# -*- coding: utf-8 -*-
"""
通用零依赖 HTTPS Keep-Alive 连接池 (`HTTPSConnectionPool`)：
带空闲 TTL 过期淘汰与 `select` 死连接预检 (Liveness Check)，自动重连。
"""

from __future__ import annotations

import http.client
import select
import socket
import ssl
import threading
import time
from typing import Dict, List, Optional, Tuple


def is_socket_alive(sock: Optional[socket.socket]) -> bool:
    """非阻塞检测底层 socket 是否已被对端关闭 (EOF) 或处于异常状态"""
    if sock is None:
        return False
    try:
        if sock.fileno() < 0:
            return False
        r, _, x = select.select([sock], [], [sock], 0.0)
        if x or r:
            # 空闲连接在未发请求前若变为可读，说明对端已发送 FIN / TLS close_notify
            return False
        return True
    except Exception:
        return False


class HTTPSConnectionPool:
    """线程安全、带 Liveness 检测与 TTL 淘汰的 Keep-Alive HTTPS 连接池"""

    def __init__(self, host: str, timeout: float = 10.0, max_pool_size: int = 20, max_idle_sec: float = 18.0):
        self.host = host
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.max_idle_sec = max_idle_sec
        self._ssl_ctx = ssl.create_default_context()
        self._pool: List[Tuple[float, http.client.HTTPSConnection]] = []
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
        now = time.monotonic()
        with self._lock:
            while self._pool:
                ts, conn = self._pool.pop()
                if (now - ts) <= self.max_idle_sec and is_socket_alive(conn.sock):
                    if timeout and conn.sock is not None:
                        try:
                            conn.sock.settimeout(timeout)
                        except Exception:
                            pass
                    return conn
                try:
                    conn.close()
                except Exception:
                    pass
        return self._create_conn(timeout=timeout)

    def _release(self, conn: http.client.HTTPSConnection) -> None:
        if conn.sock is None or not is_socket_alive(conn.sock):
            try:
                conn.close()
            except Exception:
                pass
            return
        with self._lock:
            if len(self._pool) < self.max_pool_size:
                self._pool.append((time.monotonic(), conn))
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
        fresh: bool = False,
    ) -> Tuple[int, bytes]:
        req_headers = dict(headers)
        req_headers["Connection"] = "keep-alive"
        req_headers["Content-Length"] = str(len(body))

        for attempt in range(2):
            conn = (
                self._create_conn(timeout=timeout)
                if (fresh or attempt > 0)
                else self._acquire(timeout=timeout)
            )
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
            conns = [c for _, c in self._pool]
            self._pool.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
