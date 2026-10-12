# Block or unblock distracting sites on Windows by editing the hosts file.
# Usage (elevated): powershell -ExecutionPolicy Bypass -File block-sites.ps1 on|off|status
# Sites come from ..\blocklist.txt (one domain per line).
param(
    [Parameter(Mandatory = $true)][ValidateSet('on', 'off', 'status')][string]$Action,
    [string]$ListPath = (Join-Path $PSScriptRoot '..\blocklist.txt'),
    [string]$HostsPath = (Join-Path $env:SystemRoot 'System32\drivers\etc\hosts')
)
$ErrorActionPreference = 'Stop'
$begin = '# >>> work-timer block >>>'
$end = '# <<< work-timer block <<<'

$lines = @(Get-Content -LiteralPath $HostsPath -ErrorAction SilentlyContinue)
if ($Action -eq 'status') {
    if ($lines -contains $begin) { 'on' } else { 'off' }
    return
}

$out = New-Object System.Collections.Generic.List[string]
$skip = $false
foreach ($l in $lines) {
    if ($l -eq $begin) { $skip = $true; continue }
    if ($l -eq $end) { $skip = $false; continue }
    if (-not $skip) { $out.Add($l) }
}

if ($Action -eq 'on') {
    $out.Add($begin)
    foreach ($raw in Get-Content -LiteralPath $ListPath) {
        $d = $raw.Trim()
        if (-not $d -or $d.StartsWith('#')) { continue }
        $d = $d -replace '^www\.', ''
        foreach ($h in @($d, "www.$d")) {
            $out.Add("0.0.0.0 $h")
            $out.Add("::1 $h")
        }
    }
    $out.Add($end)
}

Set-Content -LiteralPath $HostsPath -Value $out -Encoding ASCII
if ($IsWindows -ne $false) { ipconfig /flushdns | Out-Null }
"Sites " + $(if ($Action -eq 'on') { 'blocked' } else { 'unblocked' })
