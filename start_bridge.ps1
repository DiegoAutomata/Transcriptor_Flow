Param()

$HOST_ADDR = "127.0.0.1"
$PORT = 19876
$POLL_MS = 50
$PREVIEW_MS = 200

$Host.UI.RawUI.WindowTitle = "Transcriptor Flow Bridge"

# Compile Win32 API access
$typeCode = @'
using System;
using System.Runtime.InteropServices;
public class Kb {
    [DllImport("user32.dll")]
    public static extern short GetAsyncKeyState(int vKey);
}
'@
Add-Type -TypeDefinition $typeCode
Add-Type -AssemblyName System.Windows.Forms

$wshell = New-Object -ComObject "WScript.Shell"

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

function Inject-Full($text) {
    if (-not $text -or $text.Trim().Length -eq 0) {
        return
    }
    try {
        $text | clip.exe
        Start-Sleep -Milliseconds 30
        [System.Windows.Forms.SendKeys]::SendWait("^a")
        Start-Sleep -Milliseconds 30
        [System.Windows.Forms.SendKeys]::SendWait("^v")
    } catch {
        Write-Host "  Inject ERROR: $_" -ForegroundColor Red
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Transcriptor Flow Bridge v6.2" -ForegroundColor White
Write-Host "  Hold Ctrl+Alt to dictate - live preview" -ForegroundColor Gray
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
    Start-Sleep -Seconds 1
}
if (-not $connected) {
    Write-Host ""
    Write-Host "  WARNING: Could not connect to daemon at ${HOST_ADDR}:${PORT}" -ForegroundColor Yellow
    Write-Host "  Start it with: systemctl --user start transcriptor-flow" -ForegroundColor Gray
    Write-Host ""
}

$VK_CONTROL = 0x11
$VK_MENU    = 0x12
$wasActive = $false
$lastPreviewText = ""
$nextPreview = (Get-Date).AddDays(-1)

while ($true) {
    $ctrl = IsPressed $VK_CONTROL
    $alt  = IsPressed $VK_MENU
    $active = ($ctrl) -and ($alt)

    if ($active -and (-not $wasActive)) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] Ctrl+Alt -> recording" -ForegroundColor Red
        $r = Send-Tcp "start"
        if ($r -match "ok") {
            Write-Host "  OK" -ForegroundColor Green
        }
        else {
            Write-Host "  FAILED" -ForegroundColor Red
        }
        $lastPreviewText = ""
        $nextPreview = (Get-Date).AddMilliseconds($PREVIEW_MS)
    }
    elseif ((-not $active) -and $wasActive) {
        $time = Get-Date -Format "HH:mm:ss"
        Write-Host "[$time] Ctrl+Alt released -> transcribing..." -ForegroundColor Gray
        $r = Send-Tcp "stop"
        if ($r -match "^text: (.*)") {
            $texto = $matches[1]
            if ($texto) {
                Write-Host "  Text: $texto" -ForegroundColor Green
                Inject-Full $texto
            }
            else {
                Write-Host "  (nothing transcribed)" -ForegroundColor DarkGray
            }
        }
        else {
            Write-Host "  FAILED: $r" -ForegroundColor Red
        }
        $lastPreviewText = ""
    }
    elseif ($active -and $wasActive) {
        $now = Get-Date
        if ($now -ge $nextPreview) {
            $r = Send-Tcp "preview"
            if ($r -match "^text: (.*)") {
                $texto = $matches[1]
                if ($texto) {
                    if ($texto -ne $lastPreviewText) {
                        Inject-Full $texto
                        $lastPreviewText = $texto
                    }
                }
            }
            $nextPreview = (Get-Date).AddMilliseconds($PREVIEW_MS)
        }
    }

    $wasActive = $active
    Start-Sleep -Milliseconds $POLL_MS
}
