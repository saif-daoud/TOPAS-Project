$ErrorActionPreference = "Stop"
$websiteRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath (Join-Path $websiteRoot "frontend")
npm.cmd run dev

