$ErrorActionPreference = "Stop"
$websiteRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $websiteRoot
Set-Location -LiteralPath $workspaceRoot
python -m uvicorn website.backend.app:app --host 127.0.0.1 --port 8000 --reload

