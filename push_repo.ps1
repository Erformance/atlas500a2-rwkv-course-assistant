# Push this repo to GitHub from Windows (ASCII-only on purpose: PowerShell 5.1
# reads scripts without a BOM as ANSI, and non-ASCII comments break parsing).
#
# Why this exists: credential.helper=manager only pops a GUI dialog in
# non-interactive sessions; if the dialog is cancelled git dies with
# "could not read Username" (hit on 2026-09-24). Here we instead:
#   1) read the stored GitHub token via `git credential fill`;
#   2) push with http.<url>.extraheader Basic auth, so the token never lands in
#      the URL or in git's error output (and is masked in this script's output).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File push_repo.ps1                  # push HEAD
#   powershell -ExecutionPolicy Bypass -File push_repo.ps1 "commit message" # commit + push

param(
  [string]$Message = "",
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$env:GIT_TERMINAL_PROMPT = "0"
Set-Location $PSScriptRoot

if ($Message) {
  git add -A
  git -c user.name="Codex" -c user.email="codex@local" commit -q -m $Message
  if ($LASTEXITCODE -ne 0) { Write-Output "nothing to commit, pushing anyway" }
}

$cred = ("protocol=https`nhost=github.com`n`n" | git credential fill) 2>$null
if (-not $cred) {
  Write-Output "no GitHub credential found; run 'git push' once by hand to log in"
  exit 1
}
$user = (($cred | Where-Object { $_ -like 'username=*' }) -replace '^username=', '')
$pass = (($cred | Where-Object { $_ -like 'password=*' }) -replace '^password=', '')
$b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(("{0}:{1}" -f $user, $pass)))

Write-Output ("credential: {0} (token length {1})" -f $user, $pass.Length)

git -c "http.https://github.com/.extraheader=Authorization: Basic $b64" `
    push origin ("HEAD:{0}" -f $Branch) 2>&1 | ForEach-Object {
      ($_ -replace [regex]::Escape($b64), '***') -replace [regex]::Escape($pass), '***'
    }

Write-Output "remote branch now points at:"
git ls-remote origin -h ("refs/heads/{0}" -f $Branch)
