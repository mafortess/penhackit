import os
from pathlib import Path

def launch_kb_monitor_window_windows(session_dir: Path, cols: int = 60, rows: int = 14) -> None:
    """
    Opens a separate small PowerShell window that continuously shows kb.json.
    Windows-only. No interaction with the core; read-only.
    """
    if os.name != "nt":
        return
    session_dir = session_dir.resolve()
    kb_path = (session_dir / "kb.json").resolve()
    ps1 = (session_dir / "_kb_monitor.ps1").resolve()

    # # PowerShell script to set window size and refresh output
    ps_script = rf"""
    $kb = '{str(kb_path)}'
    $ErrorActionPreference='Continue'
    try {{
      $raw = $Host.UI.RawUI
      $raw.WindowTitle = 'PenHackIt KB'
      $size = New-Object System.Management.Automation.Host.Size({cols},{rows})
      $raw.WindowSize = $size
      $raw.BufferSize = New-Object System.Management.Automation.Host.Size({cols}, 3000)
    }} catch {{Write-Host "ERROR: $($_.Exception.Message)"}}
    
    while ($true) {{
      Clear-Host
      if (Test-Path -LiteralPath $kb) {{
        Get-Content -LiteralPath $kb -Raw
      }} else {{
        Write-Host "Waiting for kb.json: $kb"
      }}
      Start-Sleep -Milliseconds 500
    }}
    """.strip()
        # IMPORTANT: start "" ...  (empty title), otherwise args get mis-parsed and it exits.
   
    # Launch new console window
    # ps_script = "while ($true) { Clear-Host; 'alive'; Start-Sleep -Milliseconds 500 }"
    
    ps1.write_text(
        f"$kb='{kb_path}'; while($true){{cls; if(Test-Path $kb){{gc $kb -Raw}} else {{'Waiting for kb.json'}}; sleep -m 500}}",
        encoding="utf-8",
    )

    subprocess.Popen(["cmd", "/c", "start", "", "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)])
    # subprocess.Popen(
    #     ["cmd.exe", "/c", "start", "", "powershell.exe", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.DEVNULL,
    #     stdin=subprocess.DEVNULL,
    #     creationflags=subprocess.CREATE_NEW_CONSOLE,
    # )