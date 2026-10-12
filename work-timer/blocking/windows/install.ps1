# One-time setup for site blocking on Windows. Run from an elevated PowerShell:
#   powershell -ExecutionPolicy Bypass -File install.ps1
# Creates two scheduled tasks that run block-sites.ps1 with admin rights, and
# registers the worktimer:// link so the timer page can trigger them.
#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'block-sites.ps1'
$handler = Join-Path $PSScriptRoot 'protocol-handler.ps1'
$user = "$env:USERDOMAIN\$env:USERNAME"

foreach ($a in 'on', 'off') {
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" $a"
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
    Register-ScheduledTask -TaskPath '\WorkTimer\' -TaskName "Block-$a" -Action $action `
        -Principal $principal -Settings $settings -Force | Out-Null
}

$key = 'HKCU:\Software\Classes\worktimer'
New-Item -Path "$key\shell\open\command" -Force | Out-Null
Set-Item -Path $key -Value 'URL:Work Timer'
New-ItemProperty -Path $key -Name 'URL Protocol' -Value '' -Force | Out-Null
Set-Item -Path "$key\shell\open\command" `
    -Value "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$handler`" `"%1`""

Write-Host "Installed for $user."
Write-Host "In the timer's settings press 'Windows' to fill in worktimer://block and worktimer://unblock."
Write-Host "Keep this folder where it is; the tasks point at these scripts."
