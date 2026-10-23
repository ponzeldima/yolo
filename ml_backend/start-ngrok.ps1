[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path $_ -PathType Leaf })]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$NgrokAuthtoken,

    [string]$LabelStudioUrl,

    [string]$LabelStudioApiKey,

    [ValidateRange(1, 104857600)]
    [int]$MaxImageMegabytes = 25,

    [ValidateRange(1024, 65535)]
    [int]$BackendPort = 9090
)

$ErrorActionPreference = "Stop"
$backendDirectory = $PSScriptRoot
$python = Join-Path $backendDirectory "..\fpv_detector\.venv\Scripts\python.exe"

if (-not (Test-Path $python -PathType Leaf)) {
    throw "Python environment not found at $python. Create it or update `$python in this script."
}

if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    throw "ngrok is not installed or not on PATH. Install it from https://ngrok.com/download first."
}

& $python -m pip install -r (Join-Path $backendDirectory "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Installing backend dependencies failed."
}

$env:MODEL_DIR = Split-Path -Parent (Resolve-Path $ModelPath)
$env:YOLO_MODEL = Split-Path -Leaf $ModelPath
$env:MAX_IMAGE_BYTES = $MaxImageMegabytes * 1MB

if ($LabelStudioUrl) {
    $env:LABEL_STUDIO_URL = $LabelStudioUrl
}

if ($LabelStudioApiKey) {
    $env:LABEL_STUDIO_API_KEY = $LabelStudioApiKey
}

$waitress = Start-Process -FilePath $python `
    -ArgumentList @("-m", "waitress", "--host=127.0.0.1", "--port=$BackendPort", "model:app") `
    -WorkingDirectory $backendDirectory `
    -NoNewWindow `
    -PassThru

try {
    $healthUrl = "http://127.0.0.1:$BackendPort/health"
    $ready = $false
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        if ($waitress.HasExited) {
            throw "The prediction backend exited unexpectedly. Check the console output above."
        }

        try {
            Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $ready) {
        throw "The prediction backend did not become healthy. Check the model path and Python dependencies."
    }

    & ngrok config add-authtoken $NgrokAuthtoken
    if ($LASTEXITCODE -ne 0) {
        throw "ngrok authentication failed."
    }

    Write-Host "Backend health check passed. Starting the HTTPS tunnel..."
    & ngrok http $BackendPort
}
finally {
    if (-not $waitress.HasExited) {
        Stop-Process -Id $waitress.Id
    }
}
