#!/usr/bin/env python3
"""Render the film manifest through xiaozhi-esp32-server's EdgeTTS provider."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import edge_tts


async def synthesize_text(*, text: str, voice: str, output: Path) -> None:
    # Mirrors xiaozhi-esp32-server/core/providers/tts/edge.py without
    # importing the server bootstrap, whose global CLI parser owns argv.
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(str(output))


async def render(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_path = (
        Path(args.xiaozhi_root)
        / "main/xiaozhi-server/core/providers/tts/edge.py"
    )
    if not provider_path.exists():
        raise FileNotFoundError(f"Missing xiaozhi EdgeTTS provider: {provider_path}")

    for index, scene in enumerate(manifest["scenes"]):
        output = output_dir / f"{index:02d}-{scene['key']}.mp3"
        if not args.force and output.exists() and output.stat().st_size > 0:
            continue
        voice = scene.get("speaker", manifest["defaultSpeaker"])
        print(f"Synthesizing {scene['key']} voice={voice} -> {output.name}")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                await synthesize_text(
                    text=scene["text"].strip(),
                    voice=voice,
                    output=output,
                )
                if output.exists() and output.stat().st_size > 0:
                    break
                raise RuntimeError(f"Empty TTS output: {output}")
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    raise RuntimeError(
                        f"xiaozhi EdgeTTS failed after {attempt} attempts: "
                        f"{scene['key']}"
                    ) from exc
                print(f"  retry {attempt}/3 after {type(exc).__name__}: {exc}")
                await asyncio.sleep(attempt * 2)
        if last_error is not None and (
            not output.exists() or output.stat().st_size == 0
        ):
            raise RuntimeError(f"No audio produced for {scene['key']}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiaozhi-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(render(args))


if __name__ == "__main__":
    main()
