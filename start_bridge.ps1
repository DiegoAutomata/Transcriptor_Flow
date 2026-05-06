# Transcriptor Flow Bridge - PowerShell
# Detecta Ctrl+Alt y notifica al daemon en WSL via TCP
# Recibe texto transcrito y lo inyecta en la app activa de Windows

$Host.UI.RawUI.WindowTitle = "Transcriptor Flow Bridge"

$HOST_ADDR = "127.0.0.1"
$PORT = 19876
$POLL_MS = 50

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Kb {
    [DllImport("user32.dll")]
    public static extern short GetAsyncKeyState(int vKey);
}
"@

function IsPressed($vk) {
    return ([Kb]::GetAsyncKeyState($vk) -band 0x8000) -ne 0
}

function Send-Tcp($cmd) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient($HOST_ADDR, $PORT)
        $stream = $client.GetStream()
        $writer = New-Object System.IO.StreamWriter($stream)
        $reader = New-Object System.IO.StreamReader($stream)
        $writer.WriteLine($cmd)
        $writer.Flush()
        $resp = $reader.ReadLine()
        $client.Close()
        return $resp
    } catch {
        return $null
    }
}

function Inject-Text($text) {
    if (-not $text -or $text.Trim().Length -eq 0) {
        return
    }
    try {
        $prevClip = Get-Clipboard -Raw -ErrorAction SilentlyContinue
        Set-Clipboard -Value $text
        Start-Sleep -Milliseconds 30
        $wshell.SendKeys("^v")
        Start-Sleep -Milliseconds 80
        if ($prevClip) {
            Start-Sleep -Milliseconds 50
            Set-Clipboard -Value $prevClip
        }
    } catch {
        Write-Host "  Inject fallo: $_" -ForegroundColor Red
    }
}

$wshell = New-Object -ComObject WScript.Shell

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Transcriptor Flow Bridge (PowerShell)" -ForegroundColor White
Write-Host "  Hold Ctrl+Alt to dictate in ANY app" -ForegroundColor Gray
Write-Host "  Ctrl+C to exit" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking daemon connection..." -NoNewline
$connected = $false
for ($i = 0; $i -lt 10; $i++) {
    $r = Send-Tcp "ping"
    if ($r -eq "pong") {
        Write-Host " Connected." -ForegroundColor Green
        $connected = $true
        break
    }
    Start-Sleep 1
}
if (-not $connected) {
    Write-Host ""
    Write-Host "  WARNING: Could not connect to daemon at ${HOST_ADDR}:${PORT}" -ForegroundColor Yellow
    Write-Host "  Make sure it is running: systemctl --user start transcriptor-flow" -ForegroundColor Gray
    Write-Host ""
}

$VK_CONTROL = 0x11
$VK_MENU    = 0x12
$wasActive = $false

while ($true) {
    $ctrl = IsPressed $VK_CONTROL
    $alt  = IsPressed $VK_MENU
    $active = $ctrl -and $alt

    if ($active -and -not $wasActive) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] Ctrl+Alt -> recording" -ForegroundColor Red
        $r = Send-Tcp "start"
        if ($r -match "ok") {
            Write-Host "  OK" -ForegroundColor Green
        }
        else {
            Write-Host "  FAILED" -ForegroundColor Red
        }
    }
    elseif (-not $active -and $wasActive) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] Ctrl+Alt released -> transcribing..." -ForegroundColor Gray
        $r = Send-Tcp "stop"
        if ($r -match "^text: (.*)") {
            $texto = $matches[1]
            if ($texto) {
                Write-Host "  Text: $texto" -ForegroundColor Green
                Inject-Text $texto
            } else {
                Write-Host "  (nothing transcribed)" -ForegroundColor DarkGray
            }
        }
        else {
            Write-Host "  FAILED: $r" -ForegroundColor Red
        }
    }

    $wasActive = $active
    Start-Sleep -Milliseconds $POLL_MS
}
