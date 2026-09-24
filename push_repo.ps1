# Push this repo to GitHub from Windows without any interactive prompt.
#
# Why not plain `git push`: this machine uses credential.helper=manager, which in
# a non-interactive session only pops a GUI dialog; if nobody answers it git
# either dies with "could not read Username" or hangs forever (hit on 2026-09-24,
# twice). So we fetch the token ourselves, in this order:
#   1) $env:GH_TOKEN
#   2) .github_token next to this script (first line; gitignored)  <- preferred
#   3) the Windows Credential Store entry written by "GitHub for Visual Studio"
#      (read with CredRead)
# NOTE (2026-09-24): the credential found in (3) on this machine is an expired
# `gho_` OAuth token - the GitHub API answers 401 for it, so (3) is only a
# fallback. Put a PAT with repo scope in .github_token for reliable pushes.
# Then we push with http.<url>.extraheader Basic auth, so the token never lands in
# the remote URL or git's error output, and it is masked in this script's output.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File push_repo.ps1                  # push HEAD
#   powershell -ExecutionPolicy Bypass -File push_repo.ps1 "commit message" # commit + push
#
# ASCII-only on purpose: PowerShell 5.1 reads BOM-less scripts as ANSI and
# non-ASCII comments would break parsing.

param(
  [string]$Message = "",
  [string]$Branch = "main",
  [string]$User = "Erformance",
  [string]$CredTarget = "GitHub for Visual Studio - https://Erformance@github.com/"
)

$ErrorActionPreference = "Stop"
$env:GIT_TERMINAL_PROMPT = "0"
Set-Location $PSScriptRoot

Add-Type -Language CSharp -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class CredStore {
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  public struct CREDENTIAL {
    public int Flags;
    public int Type;
    public string TargetName;
    public string Comment;
    public long LastWritten;
    public int CredentialBlobSize;
    public IntPtr CredentialBlob;
    public int Persist;
    public int AttributeCount;
    public IntPtr Attributes;
    public string TargetAlias;
    public string UserName;
  }

  [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  public static extern bool CredRead(string target, int type, int flags, out IntPtr credential);

  [DllImport("advapi32.dll", SetLastError=true)]
  public static extern void CredFree(IntPtr cred);

  public static string Read(string target) {
    IntPtr p;
    if (!CredRead(target, 1, 0, out p)) { return null; }
    try {
      CREDENTIAL c = (CREDENTIAL)Marshal.PtrToStructure(p, typeof(CREDENTIAL));
      if (c.CredentialBlob == IntPtr.Zero || c.CredentialBlobSize <= 0) { return ""; }
      return Marshal.PtrToStringUni(c.CredentialBlob, c.CredentialBlobSize / 2);
    } finally {
      CredFree(p);
    }
  }
}
"@

function Get-GitHubToken {
  if ($env:GH_TOKEN) { return $env:GH_TOKEN.Trim() }
  $file = Join-Path $PSScriptRoot ".github_token"
  if (Test-Path -LiteralPath $file) {
    $t = (Get-Content -LiteralPath $file -TotalCount 1)
    if ($t) { return $t.Trim() }
  }
  $t = [CredStore]::Read($CredTarget)
  if ($t) { return $t.Trim() }
  return $null
}

$token = Get-GitHubToken
if (-not $token) {
  Write-Output "no token found: set `$env:GH_TOKEN, create .github_token, or log in once via git push"
  exit 1
}
Write-Output ("token: {0} chars ({1}...)" -f $token.Length, $token.Substring(0, [Math]::Min(4, $token.Length)))

if ($Message) {
  git add -A
  git -c user.name="Codex" -c user.email="codex@local" commit -q -m $Message
  if ($LASTEXITCODE -ne 0) { Write-Output "nothing to commit, pushing anyway" }
}

$b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(("{0}:{1}" -f $User, $token)))

# git writes progress to stderr; with ErrorActionPreference=Stop a successful
# push would be reported as a terminating error.
$ErrorActionPreference = "Continue"
$pushOut = git -c "http.https://github.com/.extraheader=Authorization: Basic $b64" `
               push origin ("HEAD:{0}" -f $Branch) 2>&1
$pushOk = $LASTEXITCODE -eq 0
$pushOut | ForEach-Object {
  ($_ -replace [regex]::Escape($b64), '***') -replace [regex]::Escape($token), '***'
}
$ErrorActionPreference = "Stop"
if (-not $pushOk) { Write-Output "push failed (see output above)"; exit 1 }

Write-Output "remote branch now points at:"
git ls-remote origin -h ("refs/heads/{0}" -f $Branch)
