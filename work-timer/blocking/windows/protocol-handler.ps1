# Called by Windows when the browser opens worktimer://block or worktimer://unblock.
# Runs the matching scheduled task, which already has admin rights, so there is no UAC prompt.
param([string]$Url)
if ($Url -match '^worktimer://unblock') { $action = 'off' }
elseif ($Url -match '^worktimer://block') { $action = 'on' }
else { exit 1 }
schtasks.exe /run /tn "\WorkTimer\Block-$action" | Out-Null
