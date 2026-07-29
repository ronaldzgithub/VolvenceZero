#!/usr/bin/env python3
"""Generate scene-level TTS and align it to the 110-second corporate film."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "src/data/voiceover.json"
DEFAULT_XIAOZHI_ROOT = Path(
    os.environ.get(
        "XIAOZHI_SERVER_ROOT",
        "/Users/mengfu/Documents/GitHub/xiaozhi-esp32-server",
    )
)
DEFAULT_TTS_PYTHON = os.environ.get(
    "XIAOZHI_TTS_PYTHON",
    "/Users/mengfu/miniconda3/envs/xiaozhi-esp32-server/bin/python",
)
DEFAULT_RAW_DIR = ROOT / "public/voiceover/raw"
DEFAULT_OUTPUT = ROOT / "public/voiceover/volvence-corporate-film-voiceover.wav"
DEFAULT_REPORT = ROOT / "out/voiceover-report.json"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def atempo_chain(rate: float) -> str:
    factors: list[float] = []
    while rate > 2.0:
        factors.append(2.0)
        rate /= 2.0
    while rate < 0.5:
        factors.append(0.5)
        rate /= 0.5
    factors.append(rate)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def synthesize(
    *,
    manifest: dict,
    xiaozhi_root: Path,
    tts_python: str,
    raw_dir: Path,
    force: bool,
    provider: str,
) -> None:
    if not Path(tts_python).exists():
        raise FileNotFoundError(f"Missing TTS Python runtime: {tts_python}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    pending = False
    for index, scene in enumerate(manifest["scenes"]):
        target = raw_dir / f"{index:02d}-{scene['key']}.mp3"
        if force or not target.exists() or target.stat().st_size == 0:
            pending = True
    if not pending:
        print("All TTS segments exist; synthesis skipped.")
        return

    if provider == "mac-preview":
        synthesize_mac_preview(
            manifest=manifest,
            raw_dir=raw_dir,
            force=force,
        )
        return
    if provider == "xiaozhi-doubao":
        synthesize_xiaozhi_doubao(
            manifest=manifest,
            xiaozhi_root=xiaozhi_root,
            tts_python=tts_python,
            raw_dir=raw_dir,
        )
        return
    if provider == "xiaozhi-edge":
        edge_provider = (
            xiaozhi_root
            / "main/xiaozhi-server/core/providers/tts/edge.py"
        )
        if not edge_provider.exists():
            raise FileNotFoundError(
                f"Missing xiaozhi EdgeTTS provider: {edge_provider}"
            )

        run(
            [
                tts_python,
                str(ROOT / "scripts/xiaozhi-edge-tts.py"),
                "--xiaozhi-root",
                str(xiaozhi_root),
                "--manifest",
                str(DEFAULT_MANIFEST),
                "--output-dir",
                str(raw_dir),
                *(["--force"] if force else []),
            ]
        )
        return
    raise ValueError(f"Unknown voice provider: {provider}")


def synthesize_xiaozhi_doubao(
    *,
    manifest: dict,
    xiaozhi_root: Path,
    tts_python: str,
    raw_dir: Path,
) -> None:
    """Generate the formal warm male track through xiaozhi's Doubao script."""

    batch_script = xiaozhi_root / "batch_pitch_tts.py"
    if not batch_script.exists():
        raise FileNotFoundError(f"Missing xiaozhi Doubao script: {batch_script}")

    with tempfile.TemporaryDirectory(prefix="volvence_xiaozhi_scripts_") as tmp:
        input_dir = Path(tmp)
        for index, scene in enumerate(manifest["scenes"]):
            script_name = f"{index:02d}-{scene['key']}.txt"
            (input_dir / script_name).write_text(
                scene["text"].strip() + "\n",
                encoding="utf-8",
            )
        run(
            [
                tts_python,
                str(batch_script),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(raw_dir),
            ]
        )


def synthesize_mac_preview(
    *,
    manifest: dict,
    raw_dir: Path,
    force: bool,
) -> None:
    """Build a clearly-labelled local preview when remote TTS is unavailable."""

    for index, scene in enumerate(manifest["scenes"]):
        target = raw_dir / f"{index:02d}-{scene['key']}.mp3"
        if not force and target.exists() and target.stat().st_size > 0:
            continue
        with tempfile.TemporaryDirectory(prefix="volvence_mac_voice_") as tmp:
            aiff = Path(tmp) / f"{scene['key']}.aiff"
            print(f"Preview voice {scene['key']} -> {target.name}")
            run(
                [
                    "say",
                    "-v",
                    "Reed (中文（中国大陆）)",
                    "-r",
                    "205",
                    "-o",
                    str(aiff),
                    scene["text"].strip(),
                ]
            )
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(aiff),
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    str(target),
                ]
            )


def build_track(
    *,
    manifest: dict,
    raw_dir: Path,
    output: Path,
    report_path: Path,
    provider: str,
) -> None:
    fps = int(manifest["fps"])
    head_ms = int(manifest["headPaddingMs"])
    tail_ms = int(manifest["tailPaddingMs"])
    max_playback_rate = 1.42 if provider.startswith("xiaozhi-") else 1.23
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "fps": fps,
        "voiceProvider": provider,
        "maxPlaybackRate": max_playback_rate,
        "scenes": [],
    }

    with tempfile.TemporaryDirectory(prefix="volvence_corporate_mix_") as tmp:
        tmp_dir = Path(tmp)
        scene_wavs: list[Path] = []
        for index, scene in enumerate(manifest["scenes"]):
            raw = raw_dir / f"{index:02d}-{scene['key']}.mp3"
            if not raw.exists():
                raise FileNotFoundError(f"Missing raw segment: {raw}")

            normalized = tmp_dir / f"{index:02d}-{scene['key']}-source.wav"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-fflags",
                    "+discardcorrupt",
                    "-i",
                    str(raw),
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    str(normalized),
                ]
            )
            scene_seconds = (int(scene["end"]) - int(scene["start"])) / fps
            available_seconds = scene_seconds - (head_ms + tail_ms) / 1000
            raw_seconds = probe_duration(normalized)
            playback_rate = max(1.0, raw_seconds / available_seconds)
            if playback_rate > max_playback_rate:
                raise RuntimeError(
                    f"{scene['key']} requires {playback_rate:.3f}x speech. "
                    "Shorten the script instead of producing rushed narration."
                )

            scene_wav = tmp_dir / f"{index:02d}-{scene['key']}.wav"
            filters = [
                atempo_chain(playback_rate),
                "aresample=48000",
                "aformat=sample_fmts=fltp:channel_layouts=stereo",
                "loudnorm=I=-16:TP=-1.5:LRA=7",
                f"adelay={head_ms}|{head_ms}",
            ]
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-t",
                    f"{scene_seconds:.6f}",
                    "-i",
                    "anullsrc=r=48000:cl=stereo",
                    "-i",
                    str(normalized),
                    "-filter_complex",
                    (
                        f"[1:a]{','.join(filters)}[voice];"
                        "[0:a][voice]amix=inputs=2:duration=first:"
                        "dropout_transition=0[out]"
                    ),
                    "-map",
                    "[out]",
                    "-t",
                    f"{scene_seconds:.6f}",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    str(scene_wav),
                ]
            )
            scene_wavs.append(scene_wav)
            report["scenes"].append(
                {
                    "key": scene["key"],
                    "sceneSeconds": round(scene_seconds, 3),
                    "rawSeconds": round(raw_seconds, 3),
                    "availableSeconds": round(available_seconds, 3),
                    "playbackRate": round(playback_rate, 4),
                    "text": scene["text"],
                }
            )

        concat_file = tmp_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{path}'" for path in scene_wavs) + "\n",
            encoding="utf-8",
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(output),
            ]
        )

    report["output"] = str(output)
    report["outputSeconds"] = round(probe_duration(output), 3)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Voice track: {output}")
    print(f"Alignment report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--xiaozhi-root", default=str(DEFAULT_XIAOZHI_ROOT))
    parser.add_argument("--tts-python", default=DEFAULT_TTS_PYTHON)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--provider",
        choices=("xiaozhi-doubao", "xiaozhi-edge", "mac-preview"),
        default="xiaozhi-doubao",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    synthesize(
        manifest=manifest,
        xiaozhi_root=Path(args.xiaozhi_root),
        tts_python=args.tts_python,
        raw_dir=Path(args.raw_dir),
        force=args.force,
        provider=args.provider,
    )
    build_track(
        manifest=manifest,
        raw_dir=Path(args.raw_dir),
        output=Path(args.output),
        report_path=Path(args.report),
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
