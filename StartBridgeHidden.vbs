' Transcriptor Flow Bridge - Auto-start (silent)
' Boots WSL, then launches the bridge in background
Dim shell, cmd

Set shell = CreateObject("Wscript.Shell")

' Boot WSL if not running
shell.Run "wsl.exe --distribution Ubuntu echo ""WSL ready""", 0, True

' Wait for WSL to fully boot
WScript.Sleep 3000

' Launch bridge silently
cmd = "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File ""\\wsl$\Ubuntu\home\diego\Transcriptor-Flow\start_bridge.ps1"""
shell.Run cmd, 0, False
