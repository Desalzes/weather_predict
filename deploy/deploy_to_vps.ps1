param(
    [string]$SshHost = "root@95.216.159.10",
    [string]$IdentityFile = "",
    [switch]$SetupOnly,
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$remoteDir = "/opt/weather"

$sshArgs = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
if ($IdentityFile) {
    $resolvedIdentity = (Resolve-Path -LiteralPath $IdentityFile).Path
    $sshArgs += @("-i", $resolvedIdentity)
}

# Files to deploy (exclude local-only files, venv, caches)
$includes = @(
    "main.py",
    "backfill_training_data.py",
    "train_calibration.py",
    "evaluate_calibration.py",
    "config.example.json",
    "stations.json",
    "requirements.txt",
    "src/",
    "strategy/",
    "deploy/",
    "data/calibration_models/",
    "data/forecast_archive/",
    "data/station_actuals/",
    "data/paper_trades/"
)

$excludes = @(
    ".venv/",
    ".git/",
    "__pycache__/",
    "*.pyc",
    "data/hrrr_cache/",
    "data/test_runs/",
    "data/sandbox_test_dir/",
    "data/tmp*",
    "config.json",
    "config.local.json",
    "api-credentials.txt",
    "*.pem",
    "*.key",
    ".orchestrator/",
    ".claude/",
    ".codex/",
    ".agents/",
    "codex_loop/",
    "codex_task/",
    "tests/",
    "logs/",
    "live_trading/"
)

Write-Host "=== Weather Scanner VPS Deploy ==="
Write-Host "Target: $SshHost`:$remoteDir"
Write-Host ""

# 1. Ensure remote directory exists
Write-Host "[1/3] Creating remote directory..."
& ssh @sshArgs $SshHost "mkdir -p $remoteDir"
if ($LASTEXITCODE -ne 0) { throw "SSH mkdir failed" }

# 2. Rsync files
Write-Host "[2/3] Syncing files..."
$rsyncExcludes = $excludes | ForEach-Object { "--exclude=$_" }

& rsync -avz --delete `
    @rsyncExcludes `
    "$repoRoot/" `
    "${SshHost}:${remoteDir}/"

if ($LASTEXITCODE -ne 0) { throw "rsync failed with exit code $LASTEXITCODE" }

# 3. Run setup on remote (unless skipped)
if (-not $SkipSetup) {
    Write-Host "[3/3] Running remote setup..."
    & ssh @sshArgs $SshHost "bash $remoteDir/deploy/vps_setup.sh"
    if ($LASTEXITCODE -ne 0) { throw "Remote setup failed" }
}
else {
    Write-Host "[3/3] Skipping remote setup (--SkipSetup)"
}

Write-Host ""
Write-Host "=== Deploy Complete ==="
Write-Host ""
Write-Host "Quick commands:"
Write-Host "  ssh $SshHost 'systemctl start weather-scanner.timer weather-settle.timer'"
Write-Host "  ssh $SshHost 'systemctl list-timers weather-*'"
Write-Host "  ssh $SshHost 'cd $remoteDir && .venv/bin/python main.py --once'"
Write-Host "  ssh $SshHost 'journalctl -u weather-scanner --since today'"
