param(
  [string]$WesadRoot = "data/raw/WESAD",
  [switch]$Download,
  [switch]$SyntheticTest
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Checked([string]$Program, [string[]]$Arguments) {
  & $Program @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program $Arguments" }
}
if (-not (Test-Path ".venv/Scripts/python.exe")) {
  Checked "py" @("-3.11", "-m", "venv", ".venv")
}
$ResearchPython = ".venv/Scripts/python.exe"
New-Item -ItemType Directory -Force "results/logs" | Out-Null
Start-Transcript -Path "results/logs/windows-run.log" -Append
try {
  Checked $ResearchPython @("-m", "pip", "install", "-r", "requirements.txt")
  Checked $ResearchPython @("-m", "pip", "check")
  Checked $ResearchPython @("scripts/capture_environment.py")
  Checked $ResearchPython @("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v")
  if ($SyntheticTest) { Checked $ResearchPython @("scripts/synthetic_test_pipeline.py") }
  if ($Download) { Checked $ResearchPython @("scripts/download_datasets.py", "--dataset", "wesad", "--extract") }
  Checked $ResearchPython @("-m", "src.features.extract", "--root", $WesadRoot, "--out", "data/processed/wesad_features.csv", "--window", "60", "--step", "15")
  Checked $ResearchPython @("-m", "src.train", "--features", "data/processed/wesad_features.csv", "--out", "results/wesad", "--plan", "core", "--expected-subjects", "15", "--dataset-version", "WESAD_ICMI2018", "--jobs", "4")
  Checked $ResearchPython @("-m", "src.replay", "--out", "results/wesad", "--subject", "S2", "--model", "lr", "--sensors", "all", "--raw-root", $WesadRoot)
  Checked $ResearchPython @("scripts/publish_results.py", "--source", "results/wesad")
  & $ResearchPython -m pip freeze | Out-File -Encoding utf8 "results/logs/windows-requirements-lock.txt"
} finally {
  Stop-Transcript
}
