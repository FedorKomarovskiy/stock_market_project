param(
    [string]$HtmlPath = "presentation\project_slides.html",
    [string]$PdfPath = "presentation\project_slides.pdf"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SourcePath = (Resolve-Path (Join-Path $ProjectDir $HtmlPath)).Path
$TargetPath = Join-Path $ProjectDir $PdfPath

$EdgeCandidates = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
)
$Edge = $EdgeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Edge) {
    throw "Microsoft Edge was not found. Please install Edge or export the HTML manually."
}

$Uri = [System.Uri]::new($SourcePath)
Write-Host "> $Edge --headless --disable-gpu --print-to-pdf=$TargetPath $($Uri.AbsoluteUri)"
if (Test-Path $TargetPath) {
    Remove-Item $TargetPath -Force
}
& $Edge --headless --disable-gpu "--print-to-pdf=$TargetPath" $Uri.AbsoluteUri | Out-Null
Start-Sleep -Milliseconds 500
if (-not (Test-Path $TargetPath)) {
    throw "Edge finished without creating PDF: $TargetPath"
}

Write-Host "Exported: $TargetPath"
