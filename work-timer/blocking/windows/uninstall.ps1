# Removes the scheduled tasks and the worktimer:// link, and unblocks everything.
# Run from an elevated PowerShell: powershell -ExecutionPolicy Bypass -File uninstall.ps1
#Requires -RunAsAdministrator
& (Join-Path $PSScriptRoot 'block-sites.ps1') off
foreach ($a in 'on', 'off') {
    Unregister-ScheduledTask -TaskPath '\WorkTimer\' -TaskName "Block-$a" -Confirm:$false -ErrorAction SilentlyContinue
}
Remove-Item -Path 'HKCU:\Software\Classes\worktimer' -Recurse -Force -ErrorAction SilentlyContinue
Write-Host 'Uninstalled.'
