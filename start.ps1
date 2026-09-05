param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appUrl = 'http://127.0.0.1:8517'
$alreadyRunning = $false
try {
    $health = Invoke-WebRequest -Uri ($appUrl + '/_stcore/health') -UseBasicParsing -TimeoutSec 3
    $alreadyRunning = $health.StatusCode -eq 200 -and $health.Content.Trim() -eq 'ok'
} catch {
    $alreadyRunning = $false
}
if ($alreadyRunning) {
    Write-Host ('The app is already running: ' + $appUrl)
    if (-not $NoBrowser) { Start-Process $appUrl }
    return
}
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\..\work\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $workspacePython) {
    $runPython = (Resolve-Path -LiteralPath $workspacePython).Path
    $env:HF_HOME = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\work\model-cache'))
} elseif (Test-Path -LiteralPath $localPython) {
    $runPython = $localPython
} else {
    Write-Host 'Python 3.11 or 3.12 is required. Creating the app environment...'
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        & $pythonCommand.Source -m venv .venv
    } elseif ($pyCommand) {
        & $pyCommand.Source -3 -m venv .venv
    } else {
        throw 'Python was not found. Install Python 3.12 and run this file again.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment.' }
    $runPython = $localPython
}
& $runPython -c "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(n) for n in ['streamlit','sentence_transformers','transformers','torch','sacrebleu']) else 1)"
if ($LASTEXITCODE -ne 0) {
    & $runPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your connection and retry.' }
}
$headless = if ($NoBrowser) { 'true' } else { 'false' }
& $runPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517 --server.headless $headless --browser.gatherUsageStats false
