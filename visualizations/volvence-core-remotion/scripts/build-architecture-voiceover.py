#!/usr/bin/env python3
"""Generate and align the narrated architecture-film voice track.

The speech is synthesized by xiaozhi-esp32-server/batch_pitch_tts.py. This
wrapper only prepares scene scripts, constrains each segment to its Remotion
window, normalizes loudness, and concatenates the final PCM WAV.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "src/data/architectureVoiceover.json"
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
DEFAULT_RAW_DIR = ROOT / "public/architecture-deep-dive/raw-voiceover"
DEFAULT_OUTPUT = (
    ROOT / "public/architecture-deep-dive/volvence-architecture-voiceover.wav"
)
DEFAULT_REPORT = ROOT / "out/architecture-voiceover-report.json"


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


def synthesize_missing(
    *,
    manifest: dict,
    xiaozhi_root: Path,
    tts_python: str,
    raw_dir: Path,
    force: bool,
) -> None:
    batch_script = xiaozhi_root / "batch_pitch_tts.py"
    if not batch_script.exists():
        raise FileNotFoundError(f"Missing xiaozhi TTS script: {batch_script}")
    if not Path(tts_python).exists():
        raise FileNotFoundError(f"Missing xiaozhi Python runtime: {tts_python}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="volvence_architecture_tts_") as tmp:
        input_dir = Path(tmp)
        pending = 0
        for index, scene in enumerate(manifest["scenes"]):
            target = raw_dir / f"{index:02d}-{scene['key']}.mp3"
            if force or not target.exists() or target.stat().st_size == 0:
                (input_dir / f"{index:02d}-{scene['key']}.txt").write_text(
                    scene["text"].strip() + "\n",
                    encoding="utf-8",
                )
                pending += 1

        if pending == 0:
            print("All raw TTS segments already exist; synthesis skipped.")
            return

        command = [
            tts_python,
            str(batch_script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(raw_dir),
            "--speaker",
            manifest["speaker"],
        ]
        run(command)


def build_voice_track(
    *,
    manifest: dict,
    raw_dir: Path,
    output: Path,
    report_path: Path,
) -> None:
    fps = int(manifest["fps"])
    head_ms = int(manifest["headPaddingMs"])
    tail_ms = int(manifest["tailPaddingMs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "fps": fps,
        "speaker": manifest["speaker"],
        "headPaddingMs": head_ms,
        "tailPaddingMs": tail_ms,
        "scenes": [],
    }

    with tempfile.TemporaryDirectory(prefix="volvence_architecture_mix_") as tmp:
        tmp_dir = Path(tmp)
        scene_wavs: list[Path] = []

        for index, scene in enumerate(manifest["scenes"]):
            raw = raw_dir / f"{index:02d}-{scene['key']}.mp3"
            if not raw.exists():
                raise FileNotFoundError(f"Missing raw TTS segment: {raw}")

            scene_seconds = (int(scene["end"]) - int(scene["start"])) / fps
            available_seconds = scene_seconds - (head_ms + tail_ms) / 1000
            raw_seconds = probe_duration(raw)
            playback_rate = max(1.0, raw_seconds / available_seconds)
            rendered_voice_seconds = raw_seconds / playback_rate
            scene_wav = tmp_dir / f"{index:02d}-{scene['key']}.wav"

            voice_filters = [
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
                    str(raw),
                    "-filter_complex",
                    (
                        f"[1:a]{','.join(voice_filters)}[voice];"
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
                    "startFrame": scene["start"],
                    "endFrame": scene["end"],
                    "sceneSeconds": round(scene_seconds, 4),
                    "rawSeconds": round(raw_seconds, 4),
                    "availableSeconds": round(available_seconds, 4),
                    "playbackRate": round(playback_rate, 4),
                    "renderedVoiceSeconds": round(rendered_voice_seconds, 4),
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
    report["outputSeconds"] = round(probe_duration(output), 4)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Voice track: {output}")
    print(f"Alignment report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Volvence architecture-film narration with xiaozhi TTS."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--xiaozhi-root", default=str(DEFAULT_XIAOZHI_ROOT))
    parser.add_argument("--tts-python", default=DEFAULT_TTS_PYTHON)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--mix-only",
        action="store_true",
        help="Skip network synthesis and rebuild only the aligned WAV.",
    )
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"Missing required executable: {executable}")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw_dir).resolve()
    if not args.mix_only:
        synthesize_missing(
            manifest=manifest,
            xiaozhi_root=Path(args.xiaozhi_root).resolve(),
            tts_python=args.tts_python,
            raw_dir=raw_dir,
            force=args.force,
        )
    build_voice_track(
        manifest=manifest,
        raw_dir=raw_dir,
        output=Path(args.output).resolve(),
        report_path=Path(args.report).resolve(),
    )


if __name__ == "__main__":
    main()
