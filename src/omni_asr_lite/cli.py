# -*- coding: utf-8 -*-
"""
命令行工具入口 (`python -m omni_asr_lite` / `omni-asr-lite`)
"""

from __future__ import annotations

import argparse
import sys

from .client import OmniASR
from .engines import SOGOU_LANGUAGES
from .models import Segment


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="omni-asr-lite",
        description="Omni-ASR-Lite — 零依赖可插拔多引擎并发调度语音识别与豆包 LLM 流式智能整理框架",
    )
    parser.add_argument("input", nargs="*", help="输入音频或视频文件路径 (wav/mp3/m4a/mp4/m4s/flac...)")
    parser.add_argument("-o", "--output", help="输出文件路径 (可选，如 out.md / out.srt / out.txt)")
    parser.add_argument(
        "-f",
        "--format",
        choices=["txt", "srt", "vtt", "md", "json"],
        default=None,
        help="输出格式: txt | srt | vtt | md | json",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="omni",
        help="调度模式: omni (默认并发抢占+容灾+冲刺) | ensemble (同段多引擎择优) | 指定单引擎名",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="zh",
        help="识别语种/方言 (如 zh / yue / sichuan / dongbei / wuhan / tianjin / en 等 14 种)",
    )
    parser.add_argument("--list-languages", action="store_true", help="列出内置搜狗引擎支持的 14 种语种与方言")
    parser.add_argument("-w", "--workers", type=int, default=16, help="内置搜狗引擎并发线程数 (默认 16)")
    parser.add_argument("--hotwords", default="", help="逗号分隔热词列表")
    parser.add_argument("--context", default="", help="上下文提示词")
    parser.add_argument("--no-organize", action="store_true", help="跳过豆包 LLM 智能整理，直接输出 ASR 原文")
    parser.add_argument("--no-hedging", action="store_true", help="关闭长尾对冲冲刺 (Straggler Hedging)")
    parser.add_argument("--stream", action="store_true", help="启用实时流式模式")
    parser.add_argument("--no-split", action="store_true", help="长音频不按静音分段")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式，不打印中间进度")
    args = parser.parse_args()

    if args.list_languages:
        print("Omni-ASR-Lite 内置搜狗引擎支持的 14 种语种与汉语方言：")
        for code, name in SOGOU_LANGUAGES.items():
            print(f"  {code:<22s} — {name}")
        return

    if not args.input:
        parser.error("请提供音频/视频文件路径，或使用 --list-languages 查看支持语种")

    hot = [w.strip() for w in args.hotwords.split(",") if w.strip()] if args.hotwords else None
    organize = not args.no_organize

    with OmniASR() as asr:
        for src in args.input:
            if args.stream:
                with asr.create_stream(
                    language=args.language,
                    hotwords=hot,
                    context=args.context,
                    organize=organize,
                    source_name=src,
                ) as stream:
                    for pcm_chunk in asr.iter_pcm16k_chunks(src, chunk_bytes=32000):
                        for ev in stream.feed(pcm_chunk):
                            if args.quiet:
                                continue
                            if ev.is_final:
                                print(
                                    f"\r[定稿 #{ev.segment_index:02d} {ev.engine}] [{ev.timestamp}] {ev.text}"
                                    + " " * 15,
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\r[流式 #{ev.segment_index:02d} {ev.engine}] [{ev.timestamp}] {ev.text} ...",
                                    end="",
                                    flush=True,
                                )
                    for ev in stream.flush():
                        if not args.quiet and ev.is_final:
                            print(
                                f"\r[定稿 #{ev.segment_index:02d} {ev.engine}] [{ev.timestamp}] {ev.text}"
                                + " " * 15,
                                flush=True,
                            )
                res = stream.result
            else:
                def _printer(seg: Segment):
                    if not args.quiet:
                        txt = seg.text if seg.text else "(静音/BGM)"
                        print(f"[{seg.index:02d} | {seg.engine:8s}] [{seg.timestamp}] {txt}", flush=True)

                res = asr.transcribe(
                    src,
                    mode=args.mode,
                    engine_workers={"sogou": args.workers},
                    language=args.language,
                    split=not args.no_split,
                    hotwords=hot,
                    context=args.context,
                    organize=organize,
                    enable_hedging=not args.no_hedging,
                    on_segment=_printer,
                )

            if args.output and len(args.input) == 1:
                saved = res.save(args.output, fmt=args.format)
                if not args.quiet:
                    print(
                        f"\n[✓] 已保存至: {saved} (音频 {res.duration:.1f}s, 总耗时 {res.elapsed_sec:.2f}s "
                        f"[ASR流水线 {res.asr_sec:.2f}s + LLM收尾 {res.organize_tail_sec:.2f}s], "
                        f"综合加速比 {res.duration / max(0.001, res.elapsed_sec):.1f}x, 引擎统计 {res.engine_stats})"
                    )
            elif not args.stream or args.format:
                fmt = args.format or "txt"
                if not args.quiet:
                    print("=" * 60)
                if fmt == "srt":
                    print(res.to_srt())
                elif fmt == "vtt":
                    print(res.to_vtt())
                elif fmt == "json":
                    print(res.to_json())
                elif fmt == "md":
                    print(res.to_markdown())
                else:
                    print(res.text)


if __name__ == "__main__":
    main()
