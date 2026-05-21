#requires -Version 5.1
<#
.SYNOPSIS
    Start the Volvence Zero browser chat service with a real Hugging Face Qwen substrate (Windows port).

.DESCRIPTION
    Windows / PowerShell port of start_browser_chat_qwen.sh. See that file for the
    background on model sizing, cross-session memory, and the full env-var table.
    The two scripts produce identical service behavior; only the host scripting
    layer differs.

    Windows default: Einstein full vertical (bundle + LoRA) with
    Qwen/Qwen2.5-1.5B-Instruct — matches ``.\einstein.ps1`` real mode.
    For the generic companion vertical instead:
      $env:VERTICAL = 'companion'
    For richer coherence on complex VZ prompts, override MODEL_ID:
      $env:MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'   # better coherence
      $env:MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'   # recommended quality bar
    The Mac/Linux companion script (start_browser_chat_qwen.sh) uses the same defaults.

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

    Einstein figure-as-a-service demo verticals (debt #41 / Wave K rollout)
    -----------------------------------------------------------------------
    The vertical registry now exposes four Einstein entries that mirror
    the three-condition ablation harness in
    ``lifeform_domain_figure.verification.persona``:

      VERTICAL=einstein-raw     PersonaCondition.RAW. No L1 (style) /
                                L3 (grounded decoder) / L4 (scope refusal)
                                on the synthesizer. Pure base Qwen with
                                the reviewed Einstein profile.
      VERTICAL=einstein-bundle  PersonaCondition.BUNDLE. L1+L3+L4 active;
                                figure_bundle attached (disk-backed Wave K
                                artefact when reachable, else synthetic
                                fallback). NO persona-LoRA registration.
      VERTICAL=einstein-full    PersonaCondition.BUNDLE_LORA. Same as
                                einstein-bundle plus the bundle's LoRA
                                artefact is registered in the process-wide
                                PersonaLoRAPool so the synthesizer's
                                auto-activate hook fires on each turn.
      VERTICAL=einstein         Backward-compat alias for einstein-bundle.

    Disk wiring (see lifeform_service.einstein_resolver):
      EINSTEIN_BUNDLE_ROOT         default data\figure_bundles -- where
                                   ``figure-bake bake-bundle`` /
                                   ``scripts/figure_collect_einstein.sh``
                                   wrote the persisted bundle tree.
      EINSTEIN_BUNDLE_ID           optional pin to a specific bundle id;
                                   empty selects the newest manifest.
      EINSTEIN_REQUIRE_REAL_BUNDLE 1 = fail-loud when no disk bundle is
                                   reachable. Defaults to 0 so a fresh
                                   checkout still starts.

    Known limitation (today): until debt #41 lands a real PEFT-on-Qwen
    persona LoRA, the synthetic LoRA delta is zeroed by the substrate's
    LayerNorm (debt #40), so einstein-bundle and einstein-full produce
    byte-equivalent forward outputs. The user-observable demo delta is
    einstein-raw vs einstein-bundle (L4 refusal / L3 evidence pointer /
    L1 voice style), which is real today.

    Cross-session memory
    --------------------
    ALPHA_MODE defaults to 1 so the kernel binds each session to the ``userId``
    typed in the chat UI (sent as X-Alpha-User) and persists per-user durable
    memory under MEMORY_SCOPE_ROOT_DIR. Set ALPHA_MODE=0 to fall back to the
    previous anonymous, in-memory-only behavior.

.EXAMPLE
    .\start_browser_chat_qwen.ps1                                # Einstein 1.5B default

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
      MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct      # default; matches einstein.ps1 real mode
      VERTICAL=einstein-full                   # bundle + LoRA; set companion for generic chat
      DEVICE=auto                              # auto | cpu | cuda | cuda:0 | mps
      LOCAL_FILES_ONLY=0
      OPEN_BROWSER=1
      MAX_SESSIONS=256
      IDLE_EVICTION_SECONDS=1800
      ALPHA_MODE=1                             # 1 = scoped memory, 0 = anonymous
      MEMORY_SCOPE_ROOT_DIR=<repo>\.local\browser_chat_memory
      ALPHA_USERS_FILE=                        # optional JSON allowlist
      EVIDENCE_ROOT_DIR=                       # optional alpha evidence dir
      PROTOCOL_AUTOLOAD_DIR=                   # optional dir scanned at startup
                                               # for PDFs/MDs to extract into
                                               # pending candidates (requires
                                               # PROTOCOL_LLM_* below)
      PROTOCOL_AUTOLOAD_FORCE_APPROVE=0        # DEV: 1 = auto-approve scanned
                                               # candidates (skip review). Use
                                               # only on local dev / tests.
      PROTOCOL_LLM_PROVIDER=qwen               # provider preset for the uptake
                                               # LLM. Known: openai / qwen /
                                               # dashscope / vllm /
                                               # lifeform-openai-compat / custom.
                                               # Sets a sane default base_url +
                                               # model. Override with
                                               # PROTOCOL_LLM_BASE_URL /
                                               # PROTOCOL_LLM_MODEL when needed.
      PROTOCOL_LLM_BASE_URL=                   # explicit endpoint override
                                               # (default: derived from provider)
      PROTOCOL_LLM_API_KEY=                    # API key (REQUIRED for
                                               # extraction routes; without it
                                               # the uptake routes still mount
                                               # but PDF/MD/description return
                                               # 503)
      PROTOCOL_LLM_MODEL=                      # explicit model override
                                               # (default: provider's default,
                                               # e.g. qwen-plus for qwen)
      PROTOCOL_LLM_TIMEOUT_SECONDS=60
      HF_HOME=<repo>\.local\hf-cache           # where HuggingFace caches model
                                               # weights (15+ GB per 7B Qwen).
                                               # Defaults to a per-repo cache so
                                               # downloads land on whichever
                                               # drive the checkout sits on; on
                                               # this machine the repo lives on
                                               # D so the cache also goes to D.
                                               # Set $env:HF_HOME='' to fall
                                               # back to the system default at
                                               # ~\.cache\huggingface.
      PYTHON=python                            # interpreter to use
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RootDir = $PSScriptRoot
if (-not $RootDir) {
    $RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

function Resolve-ProjectPython {
    param([string]$RepoRoot)
    $venvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        return $venvPython
    }
    if ($env:PYTHON) {
        return $env:PYTHON
    }
    return 'python'
}

function Test-NvidiaGpu {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $false
    }
    $gpuList = & nvidia-smi -L 2>$null
    return ($LASTEXITCODE -eq 0 -and $gpuList)
}

function Initialize-HfDownloadEnv {
    param(
        [Parameter(Mandatory)] [string] $PythonBin,
        [Parameter(Mandatory)] [string] $ModelId,
        [Parameter(Mandatory)] [string] $HfHome
    )
    if ($env:VOLVENCE_FORCE_HF_ENDPOINT) {
        if ($env:HF_ENDPOINT) {
            Write-Host "[start-browser-chat-qwen] hf_endpoint=$($env:HF_ENDPOINT) (forced)"
        } else {
            Write-Host "[start-browser-chat-qwen] hf_endpoint=<default huggingface.co> (forced)"
        }
        return
    }

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
    $probeWithEndpoint = {
        param([string]$Endpoint, [string]$TempHome)
        if ($Endpoint) {
            $env:HF_ENDPOINT = $Endpoint
        } else {
            Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue
        }
        $env:HF_HOME = $TempHome
        New-Item -ItemType Directory -Force -Path $TempHome | Out-Null
        $env:HF_PROBE_MODEL = $ModelId
        & $PythonBin -c @"
import os, sys, warnings
warnings.filterwarnings('ignore')
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(os.environ['HF_PROBE_MODEL'], 'config.json')
except Exception:
    sys.exit(1)
"@ 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }

    $configuredEndpoint = $env:HF_ENDPOINT
    if ([string]::IsNullOrEmpty($configuredEndpoint)) {
        $tempHome = Join-Path $env:TEMP ("vz_hf_probe_{0}" -f ([Guid]::NewGuid().ToString('N')))
        try {
            if (& $probeWithEndpoint '' $tempHome) {
                Write-Host "[start-browser-chat-qwen] hf_endpoint=<default huggingface.co>"
                return
            }
        } finally {
            Remove-Item -Recurse -Force $tempHome -ErrorAction SilentlyContinue
            $env:HF_HOME = $HfHome
        }
        Write-Error @"
Cannot reach huggingface.co to download '$ModelId'.

Check network / proxy, or set a working mirror explicitly:
  `$env:HF_ENDPOINT = 'https://hf-mirror.com'
  `$env:VOLVENCE_FORCE_HF_ENDPOINT = '1'
  .\start_browser_chat_qwen.ps1
"@
        exit 1
    }

    Write-Host "[start-browser-chat-qwen] probing HF_ENDPOINT=$configuredEndpoint ..."
    $tempHome = Join-Path $env:TEMP ("vz_hf_probe_{0}" -f ([Guid]::NewGuid().ToString('N')))
    $mirrorOk = $false
    try {
        $mirrorOk = (& $probeWithEndpoint $configuredEndpoint $tempHome)
    } finally {
        Remove-Item -Recurse -Force $tempHome -ErrorAction SilentlyContinue
        $env:HF_HOME = $HfHome
    }

    if ($mirrorOk) {
        $env:HF_ENDPOINT = $configuredEndpoint
        Write-Host "[start-browser-chat-qwen] hf_endpoint=$configuredEndpoint"
        return
    }

    Write-Warning "[start-browser-chat-qwen] HF_ENDPOINT=$configuredEndpoint failed huggingface_hub probe; falling back to huggingface.co"
    Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue

    $tempHome2 = Join-Path $env:TEMP ("vz_hf_probe_{0}" -f ([Guid]::NewGuid().ToString('N')))
    try {
        if (& $probeWithEndpoint '' $tempHome2) {
            Write-Host "[start-browser-chat-qwen] hf_endpoint=<default huggingface.co> (mirror fallback)"
            return
        }
    } finally {
        Remove-Item -Recurse -Force $tempHome2 -ErrorAction SilentlyContinue
        $env:HF_HOME = $HfHome
    }

    Write-Error @"
Cannot download '$ModelId' from HF_ENDPOINT or huggingface.co.

Your shell has HF_ENDPOINT=$configuredEndpoint but the mirror is unreachable
from huggingface_hub on this machine. Fix one of:

  Remove-Item Env:HF_ENDPOINT
  .\start_browser_chat_qwen.ps1
"@
    exit 1
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

$PythonBin = Resolve-ProjectPython -RepoRoot $RootDir

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
Set-DefaultEnv 'VERTICAL'              'einstein-full'
# Einstein figure-as-a-service demo wiring (see header for details).
# Read by lifeform_service.einstein_resolver when VERTICAL is one of
# einstein / einstein-raw / einstein-bundle / einstein-full.
Set-DefaultEnv 'EINSTEIN_BUNDLE_ROOT'        (Join-Path $RootDir 'data\figure_bundles')
Set-DefaultEnv 'EINSTEIN_BUNDLE_ID'          ''
Set-DefaultEnv 'EINSTEIN_REQUIRE_REAL_BUNDLE' '0'
Set-DefaultEnv 'MODEL_ID'              'Qwen/Qwen2.5-1.5B-Instruct'
if (Test-NvidiaGpu) {
    Set-DefaultEnv 'DEVICE'            'cuda'
} else {
    Set-DefaultEnv 'DEVICE'            'auto'
}
Set-DefaultEnv 'LOCAL_FILES_ONLY'      '0'
Set-DefaultEnv 'MAX_SESSIONS'          '256'
Set-DefaultEnv 'IDLE_EVICTION_SECONDS' '1800'
Set-DefaultEnv 'OPEN_BROWSER'          '1'
Set-DefaultEnv 'ALPHA_MODE'            '1'
Set-DefaultEnv 'MEMORY_SCOPE_ROOT_DIR' (Join-Path $RootDir '.local\browser_chat_memory')
Set-DefaultEnv 'ALPHA_USERS_FILE'      ''
Set-DefaultEnv 'EVIDENCE_ROOT_DIR'     ''
Set-DefaultEnv 'TEMPLATES_ROOT_DIR'    (Join-Path $RootDir 'artifacts\lifeform-templates')
Set-DefaultEnv 'MODEL_ID_ALLOWLIST'    ''
Set-DefaultEnv 'PROTOCOL_AUTOLOAD_DIR'        ''
Set-DefaultEnv 'PROTOCOL_AUTOLOAD_FORCE_APPROVE' '0'
Set-DefaultEnv 'PROTOCOL_LLM_PROVIDER'        'qwen'
Set-DefaultEnv 'PROTOCOL_LLM_BASE_URL'        ''
Set-DefaultEnv 'PROTOCOL_LLM_API_KEY'         ''
Set-DefaultEnv 'PROTOCOL_LLM_MODEL'           ''
Set-DefaultEnv 'PROTOCOL_LLM_TIMEOUT_SECONDS' '60'

# GPU machines: treat inherited DEVICE=auto as cuda unless explicitly overridden.
if (Test-NvidiaGpu) {
    $deviceNow = [Environment]::GetEnvironmentVariable('DEVICE', 'Process')
    if ([string]::IsNullOrEmpty($deviceNow) -or $deviceNow -eq 'auto') {
        $env:DEVICE = 'cuda'
    }
}

# HuggingFace cache directory. We distinguish "unset" from "empty" so
# the operator can opt into the system default (~\.cache\huggingface)
# by setting $env:HF_HOME='' explicitly. Default = per-repo cache,
# which puts model weights on the same drive as the checkout. The
# Set-DefaultEnv helper above does NOT distinguish unset/empty, so
# we hand-roll the check here.
$existingHfHome = [Environment]::GetEnvironmentVariable('HF_HOME', 'Process')
if ($null -eq $existingHfHome) {
    $env:HF_HOME = Join-Path $RootDir '.local\hf-cache'
}
if (-not [string]::IsNullOrEmpty($env:HF_HOME)) {
    New-Item -ItemType Directory -Path $env:HF_HOME -Force | Out-Null
}

# Quiet Windows symlink warning; caching still works without symlinks.
if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
}

Set-Location $RootDir

Initialize-HfDownloadEnv -PythonBin $PythonBin -ModelId $env:MODEL_ID -HfHome $env:HF_HOME

# Fail fast with a helpful hint when the chosen interpreter lacks deps.
$preflight = & $PythonBin -c @"
import aiohttp, transformers, torch
print('deps ok', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"@ 2>&1
if ($LASTEXITCODE -ne 0) {
    $hint = @"
Python at '$PythonBin' cannot load the HF stack required for real Qwen chat.

Preflight output:
$preflight

Fix (recommended on Windows — avoid conda-created venvs):
  Remove-Item -Recurse -Force .venv
  `$py = `"`$env:LOCALAPPDATA\Programs\Python\Python311\python.exe`"
  & `$py -m venv .venv
  .\install.ps1 -PythonBin .\.venv\Scripts\python.exe -Extras hf

This machine has an NVIDIA GPU — install.ps1 will pull torch from cu126 automatically.
If torch fails with WinError 1114 / c10.dll, recreate .venv with python.org first.

Then retry:
  .\start_browser_chat_qwen.ps1
"@
    Write-Error $hint
    exit 1
}

if (Test-NvidiaGpu) {
    & $PythonBin -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error @"
NVIDIA GPU detected but torch.cuda.is_available() is False.

Reinstall the CUDA wheel:
  .\.venv\Scripts\python.exe -m pip uninstall -y torch
  .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126

Or rerun:
  .\install.ps1 -PythonBin .\.venv\Scripts\python.exe -Extras hf
"@
        exit 1
    }
}

# Optional: source secrets from a non-committed .local/llm.env.ps1
# so operators don't have to paste the API key every shell. The
# .local/ tree is gitignored. The script only reads it; it never
# writes secrets back. Format inside the file:
#   $env:PROTOCOL_LLM_API_KEY = 'sk-...'
#   $env:PROTOCOL_LLM_PROVIDER = 'qwen'   # optional override
$llmEnvFile = Join-Path $RootDir '.local\llm.env.ps1'
if (Test-Path $llmEnvFile) {
    . $llmEnvFile
}

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
Write-Host "[start-browser-chat-qwen] python=$PythonBin"
Write-Host "[start-browser-chat-qwen] device=$($env:DEVICE) local_files_only=$($env:LOCAL_FILES_ONLY)"
if (Test-NvidiaGpu) {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    Write-Host "[start-browser-chat-qwen] gpu=$gpuName"
}
$hfHomeLabel = if ([string]::IsNullOrEmpty($env:HF_HOME)) { '<system default>' } else { $env:HF_HOME }
Write-Host "[start-browser-chat-qwen] hf_home=$hfHomeLabel"
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
from lifeform_service.openai_compat_client import (
    build_client_from_env,
    describe_active_provider,
)
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from lifeform_service.substrate_registry import (
    build_qwen_runtime_loader,
    build_substrate_provider_from_env,
)
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

    default_vertical = requested_vertical or default_vertical_name()
    if default_vertical not in verticals:
        print(
            f"Unknown VERTICAL={default_vertical!r}. Available: {sorted(verticals)}",
            file=sys.stderr,
        )
        return 1

    alpha_config = _build_alpha_config()
    if alpha_config.enabled and verticals[default_vertical].alpha_factory is None:
        print(
            f"default vertical {default_vertical!r} does not support alpha mode; "
            "pick a different VERTICAL or set ALPHA_MODE=0.",
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

    runtime_loader = build_qwen_runtime_loader(
        device=device,
        local_files_only=local_files_only,
        fallback_mode=SubstrateFallbackMode.DENY,
    )
    substrate_provider = build_substrate_provider_from_env(
        initial_runtime=runtime,
        initial_model_id=model_id,
        runtime_loader=runtime_loader,
        allowlist_env=_env_str_or_none("MODEL_ID_ALLOWLIST"),
    )

    templates_root_dir = _env_str_or_none("TEMPLATES_ROOT_DIR")

    # Packet 9.x engineering wrap: build the protocol uptake
    # service so PDF / MD / task-description / API-injection HTTP
    # routes are mounted. Extraction routes need an
    # OpenAI-compatible LLM (PROTOCOL_LLM_*); when unset, the
    # service still mounts and the routes report 503 with a
    # configuration hint.
    autoload_dir_raw = _env_str_or_none("PROTOCOL_AUTOLOAD_DIR")
    autoload_force_approve = _env_bool("PROTOCOL_AUTOLOAD_FORCE_APPROVE", default=False)

    # ONE shared external-LLM client for all consumers
    # (protocol uptake AND any vertical that opts in via
    # app["external_llm_client"]). Same provider config / API key /
    # quota across consumers — operators control external-LLM
    # spend in one place.
    shared_llm_client = build_client_from_env()

    uptake_service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=Path(autoload_dir_raw) if autoload_dir_raw else None,
            autoload_force_approve=autoload_force_approve,
            llm_client_factory=lambda: shared_llm_client,
        ),
    )

    app = create_app(
        verticals=verticals,
        default_vertical=default_vertical,
        max_sessions=max_sessions,
        idle_eviction_seconds=idle_eviction_seconds,
        substrate_provider=substrate_provider,
        alpha_config=alpha_config,
        templates_root_dir=templates_root_dir,
        protocol_uptake_service=uptake_service,
        external_llm_client=shared_llm_client,
    )

    if uptake_service._config.autoload_dir is not None:
        async def _run_autoload(_: web.Application) -> None:
            results = await uptake_service.autoload_directory()
            ok = sum(1 for r in results if r.status == "ok")
            err = sum(1 for r in results if r.status == "error")
            print(
                f"[start-browser-chat-qwen] protocol autoload: "
                f"dir={uptake_service._config.autoload_dir} "
                f"ok={ok} error={err} "
                f"(force_approve={uptake_service._config.autoload_force_approve})",
                flush=True,
            )
        app.on_startup.append(_run_autoload)
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
    available_models = ", ".join(
        spec.model_id for spec in substrate_provider.available
    )
    print(
        "[start-browser-chat-qwen] substrate-swap ENABLED "
        f"current={substrate_provider.current_model_id} "
        f"allowlist=[{available_models}]",
        flush=True,
    )
    provider_info = describe_active_provider()
    if uptake_service.llm_client is not None:
        print(
            "[start-browser-chat-qwen] protocol uptake routes ENABLED "
            f"provider={provider_info['provider']} "
            f"base_url={provider_info['base_url']} "
            f"model={provider_info['model']}",
            flush=True,
        )
    else:
        print(
            "[start-browser-chat-qwen] protocol uptake routes mounted "
            "but extraction is disabled "
            f"(provider={provider_info['provider']}, api_key_present=no). "
            "Set PROTOCOL_LLM_API_KEY (and optionally PROTOCOL_LLM_PROVIDER) "
            "to enable PDF/MD/description routes; from-payload still works.",
            flush=True,
        )
    available_verticals = ", ".join(sorted(verticals))
    print(
        "[start-browser-chat-qwen] ready "
        f"default_vertical={default_vertical} "
        f"verticals=[{available_verticals}] "
        f"model_id={model_id} runtime_origin={runtime_origin}",
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
