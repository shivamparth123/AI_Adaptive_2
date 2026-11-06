<#
 setup.ps1 - one-time setup on a new Windows PC (PowerShell)

 Run from the project folder:
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     .\setup.ps1                         # virtual environment + Python packages + checks
     .\setup.ps1 -PullModels             # ... and download the 5 required models (about 14 GB)
     .\setup.ps1 -SetOllamaEnv           # ... and set OLLAMA_MAX_LOADED_MODELS=1 (recommended)
     .\setup.ps1 -PullModels -WithBenchmarkModel   # + deepseek-r1:8b (5.2 GB), only to re-run the benchmark
 You can combine the switches. It is safe to run again.
#>
param(
    [switch]$PullModels,
    [switch]$WithBenchmarkModel,
    [switch]$SetOllamaEnv
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

Step "Checking Python"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "Python was not found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), open a NEW PowerShell and run this script again." -ForegroundColor Red
    exit 1
}
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "Found Python $ver"
if ([version]$ver -lt [version]"3.10") {
    Write-Host "Python 3.10 or newer is required (3.11 recommended)." -ForegroundColor Red
    exit 1
}

Step "Creating the virtual environment (venv)"
if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    python -m venv venv
} else {
    Write-Host "venv already exists - reusing it"
}
$venvPy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

Step "Installing Python packages (needs internet)"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt

Step "Creating data folders"
New-Item -ItemType Directory -Force -Path "logs", "webapp\backend\uploads" | Out-Null

Step "Checking Ollama"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "Ollama was not found. Install it from https://ollama.com/download, open a NEW PowerShell and run this script again." -ForegroundColor Yellow
} else {
    & ollama --version
    if ($SetOllamaEnv) {
        [Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
        Write-Host "Set OLLAMA_MAX_LOADED_MODELS=1 for your user. Quit Ollama from the tray icon, start it again, and open a NEW terminal." -ForegroundColor Yellow
    }
    if ($PullModels) {
        $models = @("llama3.2:3b", "phi4-mini", "qwen3:4b", "gemma3:4b", "qwen3-vl:4b")
        if ($WithBenchmarkModel) { $models += "deepseek-r1:8b" }
        foreach ($m in $models) {
            Step "Downloading model $m (this can take a while)"
            & ollama pull $m
        }
    } else {
        Write-Host "Models were not downloaded (add -PullModels). The list is in README.md."
    }
}

Step "Running the setup check"
& $venvPy tools\check_setup.py
Write-Host "`nDone. Start the web app with run_web.bat or the command-line app with run_cli.bat." -ForegroundColor Green
