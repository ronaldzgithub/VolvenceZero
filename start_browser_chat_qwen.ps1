#requires -Version 5.1
<#
.SYNOPSIS
    Start the Volvence Zero browser chat service with a real Hugging Face Qwen substrate (Windows port).

.DESCRIPTION
    Windows / PowerShell port of start_browser_chat_qwen.sh. See that file for the
    background on model sizing, cross-session memory, and the full env-var table.
    The two scripts produce identical service behavior; only the host scripting
    layer differs.

    Defaults target Qwen2.5-7B-Instruct: the smallest base model that reliably
    follows VZ's structured system prompt AND keeps multi-turn coherence on
    short follow-ups. The 0.5B / 1.5B / 3B variants tend to collapse into
    single-character or off-topic replies once the kernel's plan/ordering
    instructions are stacked on top of the user turn.

    What you can actually run locally
    ---------------------------------
    Reference target: MacBook Air M4, 24 GB unified memory, ~30 GB free disk.
    bf16 = transformers default, Q4 = GGUF / llama.cpp 4-bit quantization.

      Model                       bf16 RAM   Q4 RAM   Disk    Verdict on M4 24GB
      --------------------------- ---------- -------- ------- ------------------
      Qwen2.5-1.5B-Instruct       ~ 4 GB     ~ 1 GB    3 GB   too weak for VZ prompt
      Qwen2.5-3B-Instruct         ~ 8 GB     ~ 2 GB    6 GB   borderline coherent
      Qwen2.5-7B-Instruct         ~16 GB     ~ 5 GB   15 GB   recommended (default)
      Qwen2.5-14B-Instruct        ~28 GB     ~ 9 GB   28 GB   bf16 NO; Q4 OK
      Qwen2.5-32B-Instruct        ~64 GB     ~18 GB   62 GB   bf16 NO; Q4 tight
      Qwen2.5-72B-Instruct       ~145 GB     ~40 GB  145 GB   NOT runnable locally
      Qwen3-235B-A22B (MoE)      ~470 GB    ~120 GB  470 GB   NOT runnable locally
      Qwen3-Coder-480B-A35B (MoE)~960 GB    ~150 GB  960 GB   NOT runnable locally

    On a typical Windows workstation (32 GB DDR5, RTX 4080 16 GB), Qwen2.5-7B-Instruct
    in bf16 on CPU is the comfortable default; for GPU acceleration set DEVICE=cuda
    and the 7B / 14B variants both fit on a 16 GB card with bf16 / Q4 respectively.

    Cross-session memory
    --------------------
    ALPHA_MODE defaults to 1 so the kernel binds each session to the ``userId``
    typed in the chat UI (sent as X-Alpha-User) and persists per-user durable
    memory under MEMORY_SCOPE_ROOT_DIR. Set ALPHA_MODE=0 to fall back to the
    previous anonymous, in-memory-only behavior.

.EXAMPLE
    .\start_browser_chat_qwen.ps1                                # 7B default

.EXAMPLE
    $env:MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
    .\start_browser_chat_qwen.ps1

.EXAMPLE
    # If HuggingFace is slow / blocked, route through the mirror:
    $env:HF_ENDPOINT = 'https://hf-mirror.com'
    .\start_browser_chat_qwen.ps1

.EXAMPLE
    $env:ALPHA_MODE = '0'   # anonymous, no persistence
    .\start_browser_chat_qwen.ps1

.NOTES
    Useful env vars (all optional, defaults shown):
      HOST=127.0.0.1
      PORT=8765
      MODEL_ID=Qwen/Qwen2.5-7B-Instruct        # see sizing table above
      DEVICE=auto                              # auto | cpu | cuda | cuda:0 | mps
      LOCAL_FILES_ONLY=0
      OPEN_BROWSER=1
      MAX_SESSIONS=256
      IDLE_EVICTION_SECONDS=1800
      ALPHA_MODE=1                             # 1 = scoped memory, 0 = anonymous
      MEMORY_SCOPE_ROOT_DIR=<repo>\.local\browser_chat_memory
      ALPHA_USERS_FILE=                        # optional JSON allowlist
      EVIDENCE_ROOT_DIR=                       # optional alpha evidence dir
      PYTHON=python                            # interpreter to use
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RootDir = $PSScriptRoot
if (-not $RootDir) {
    $RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

# Honor existing env-var values; only fill in defaults when missing or empty.
function Set-DefaultEnv {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [AllowEmptyString()] [string] $Value
    )
    $current = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrEmpty($current)) {
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

Set-DefaultEnv 'HOST'                  '127.0.0.1'
Set-DefaultEnv 'PORT'                  '8765'
Set-DefaultEnv 'VERTICAL'              'companion'
Set-DefaultEnv 'MODEL_ID'              'Qwen/Qwen2.5-7B-Instruct'
Set-DefaultEnv 'DEVICE'                'auto'
Set-DefaultEnv 'LOCAL_FILES_ONLY'      '0'
Set-DefaultEnv 'MAX_SESSIONS'          '256'
Set-DefaultEnv 'IDLE_EVICTION_SECONDS' '1800'
Set-DefaultEnv 'OPEN_BROWSER'          '1'
Set-DefaultEnv 'ALPHA_MODE'            '1'
Set-DefaultEnv 'MEMORY_SCOPE_ROOT_DIR' (Join-Path $RootDir '.local\browser_chat_memory')
Set-DefaultEnv 'ALPHA_USERS_FILE'      ''
Set-DefaultEnv 'EVIDENCE_ROOT_DIR'     ''
Set-DefaultEnv 'TEMPLATES_ROOT_DIR'    (Join-Path $RootDir 'artifacts\lifeform-templates')

Set-Location $RootDir

# Build PYTHONPATH from packages\*\src — Windows uses ';' as the separator.
$packageSrcs = Get-ChildItem -Path (Join-Path $RootDir 'packages') -Directory -ErrorAction Stop |
    ForEach-Object { Join-Path $_.FullName 'src' } |
    Where-Object { Test-Path $_ }
$packagePaths = ($packageSrcs -join ';')
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$packagePaths;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $packagePaths
}

# Port-in-use check. Get-NetTCPConnection ships with Windows 8+ / Server 2012+.
# It raises ``CmdletizationQuery_NotFound,Get-NetTCPConnection`` when no
# listener exists on the requested port (the happy path); we match on the
# locale-independent FullyQualifiedErrorId so this works on non-English
# Windows installs as well.
$portNum = [int]$env:PORT
$listener = $null
try {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $portNum -ErrorAction Stop |
        Select-Object -First 1
} catch {
    if ($_.FullyQualifiedErrorId -notlike 'CmdletizationQuery_NotFound*') {
        throw
    }
}
if ($listener) {
    Write-Error "Port $portNum is already in use (pid=$($listener.OwningProcess)). Stop the existing service or set `$env:PORT to another port."
    exit 1
}

$chatUrl = "http://$($env:HOST):$($env:PORT)/chat"

Write-Host "[start-browser-chat-qwen] model=$($env:MODEL_ID)"
Write-Host "[start-browser-chat-qwen] device=$($env:DEVICE) local_files_only=$($env:LOCAL_FILES_ONLY)"
Write-Host "[start-browser-chat-qwen] url=$chatUrl"

if ($env:OPEN_BROWSER -eq '1') {
    # Spawn a detached helper PowerShell that waits for the server to come
    # up, then opens the chat URL with the user's default browser. Hidden
    # window so it does not steal focus.
    $launcher = "Start-Sleep -Seconds 5; Start-Process '$chatUrl'"
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden', '-Command', $launcher `
        -WindowStyle Hidden | Out-Null
}

# Materialize the Python entrypoint to a temp file so that PowerShell's pipe
# encoding (UTF-16 by default in 5.1) cannot corrupt the bytes Python reads
# from stdin. The temp file is removed even if the server is interrupted.
$pythonCode = @'
from __future__ import annotations

import os
import sys
from pathlib import Path

from aiohttp import web
from lifeform_service.alpha import (
    AlphaServiceConfig,
    load_alpha_users,
)
from lifeform_service.app import create_app
from lifeform_service.verticals import default_vertical_name, discover_verticals
from volvence_zero.substrate import SubstrateFallbackMode, build_transformers_runtime_with_fallback


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str_or_none(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
    return raw or None


def _build_alpha_config() -> AlphaServiceConfig:
    """Compose the AlphaServiceConfig from environment variables.

    Defaults: alpha mode ON, no allowlist (empty alpha_users = accept any
    user_id), memory_scope_root_dir under ./.local/browser_chat_memory.
    These defaults give the browser path cross-session memory: the
    chat UI binds the typed ``userId`` to a UserIdentity, which the
    kernel uses to build a filesystem-backed scoped MemoryStore so
    rupture-repair durable entries survive across sessions.
    """

    if not _env_bool("ALPHA_MODE", default=True):
        return AlphaServiceConfig()
    memory_dir = _env_str_or_none("MEMORY_SCOPE_ROOT_DIR")
    if memory_dir is None:
        raise RuntimeError(
            "ALPHA_MODE=1 requires MEMORY_SCOPE_ROOT_DIR to be set."
        )
    Path(memory_dir).mkdir(parents=True, exist_ok=True)
    evidence_dir = _env_str_or_none("EVIDENCE_ROOT_DIR")
    if evidence_dir is not None:
        Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    alpha_users_file = _env_str_or_none("ALPHA_USERS_FILE")
    return AlphaServiceConfig(
        enabled=True,
        memory_scope_root_dir=memory_dir,
        evidence_root_dir=evidence_dir,
        alpha_users=load_alpha_users(alpha_users_file),
    )


def main() -> int:
    host = os.environ["HOST"]
    port = int(os.environ["PORT"])
    requested_vertical = os.environ.get("VERTICAL")
    model_id = os.environ["MODEL_ID"]
    device = os.environ["DEVICE"]
    local_files_only = _env_bool("LOCAL_FILES_ONLY")
    max_sessions = int(os.environ["MAX_SESSIONS"])
    idle_eviction_seconds = float(os.environ["IDLE_EVICTION_SECONDS"])

    verticals = discover_verticals()
    if not verticals:
        print(
            "No verticals available. Install lifeform-domain-emogpt or another lifeform-domain-* package.",
            file=sys.stderr,
        )
        return 1

    vertical_name = requested_vertical or default_vertical_name()
    if vertical_name not in verticals:
        print(f"Unknown vertical {vertical_name!r}. Available: {sorted(verticals)}", file=sys.stderr)
        return 1

    alpha_config = _build_alpha_config()
    if alpha_config.enabled and verticals[vertical_name].alpha_factory is None:
        print(
            f"vertical {vertical_name!r} does not support alpha mode; "
            "set ALPHA_MODE=0 to fall back to anonymous in-memory sessions.",
            file=sys.stderr,
        )
        return 1

    print("[start-browser-chat-qwen] loading real Qwen substrate; fallback is disabled", flush=True)
    runtime = build_transformers_runtime_with_fallback(
        model_id=model_id,
        device=device,
        local_files_only=local_files_only,
        fallback_mode=SubstrateFallbackMode.DENY,
        allow_live_substrate_mutation=False,
    )
    runtime_origin = getattr(runtime, "runtime_origin")
    if runtime_origin == "builtin-fallback":
        raise RuntimeError("Expected a real HF Qwen runtime, got builtin-fallback.")

    templates_root_dir = _env_str_or_none("TEMPLATES_ROOT_DIR")
    app = create_app(
        vertical=verticals[vertical_name],
        max_sessions=max_sessions,
        idle_eviction_seconds=idle_eviction_seconds,
        substrate_runtime=runtime,
        alpha_config=alpha_config,
        templates_root_dir=templates_root_dir,
    )
    if alpha_config.enabled:
        allowlist_size = len(alpha_config.alpha_users)
        allowlist_label = (
            f"allowlist={allowlist_size}-users"
            if allowlist_size
            else "allowlist=open"
        )
        print(
            "[start-browser-chat-qwen] cross-session memory ENABLED "
            f"memory_scope_root_dir={alpha_config.memory_scope_root_dir} "
            f"{allowlist_label}",
            flush=True,
        )
        print(
            "[start-browser-chat-qwen] type a 'userId' in the chat UI to "
            "identify yourself; the kernel binds it to a per-user MemoryStore.",
            flush=True,
        )
    else:
        print(
            "[start-browser-chat-qwen] anonymous mode (ALPHA_MODE=0); "
            "no cross-session memory, no per-user scope.",
            flush=True,
        )
    if templates_root_dir is not None:
        print(
            "[start-browser-chat-qwen] templates ENABLED "
            f"templates_root_dir={templates_root_dir}",
            flush=True,
        )
    else:
        print(
            "[start-browser-chat-qwen] templates DISABLED "
            "(set TEMPLATES_ROOT_DIR to enable list/save in chat UI)",
            flush=True,
        )
    print(
        "[start-browser-chat-qwen] ready "
        f"vertical={vertical_name} model_id={model_id} runtime_origin={runtime_origin}",
        flush=True,
    )
    print(f"[start-browser-chat-qwen] listening on http://{host}:{port}/chat", flush=True)
    web.run_app(app, host=host, port=port, print=lambda *_: None)
    return 0


raise SystemExit(main())
'@

$pyTmpDir = Join-Path $env:TEMP 'volvence_zero'
New-Item -ItemType Directory -Path $pyTmpDir -Force | Out-Null
$pyTmpFile = Join-Path $pyTmpDir ("start_browser_chat_qwen_{0}.py" -f ([Guid]::NewGuid().ToString('N')))
# Write as UTF-8 without BOM so the Python parser does not see a stray byte.
[System.IO.File]::WriteAllText(
    $pyTmpFile,
    $pythonCode,
    (New-Object System.Text.UTF8Encoding $false)
)

try {
    & $PythonBin $pyTmpFile
    $exitCode = $LASTEXITCODE
} finally {
    Remove-Item -Path $pyTmpFile -Force -ErrorAction SilentlyContinue
}

exit $exitCode
