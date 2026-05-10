' Transcriptor Flow Bridge - Auto-start (silent)
' Boots WSL, then launches the bridge in background
Dim shell, cmd

Set shell = CreateObject("Wscript.Shell")

' Boot WSL and keep it alive so systemd daemon doesn't get killed
shell.Run "wsl.exe --distribution Ubuntu bash -c ""sleep infinity""", 0, False

' Wait for WSL to fully boot systemd services
WScript.Sleep 5000

' Launch bridge silently
cmd = "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File ""\\wsl$\Ubuntu\home\diego\Transcriptor-Flow\start_bridge.ps1"""
shell.Run cmd, 0, False
