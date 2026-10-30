# Circuit breaker: stop this project's own GPU/model jobs if system commit headroom gets dangerously low (a crash would waste far more than a restart).
$sp = "<scratch>"
$log = "$sp\guard_log.txt"
$limitGB = 1.0
"$(Get-Date -Format 'HH:mm:ss') guard started, trip below $limitGB GB commit headroom" | Out-File $log -Append -Encoding utf8
$end = (Get-Date).AddHours(12)
while ((Get-Date) -lt $end) {
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $free = $os.FreeVirtualMemory / 1MB
        if ($free -lt $limitGB) {
            $jobs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and ($_.CommandLine -match '_worker\.py|ipykernel_launcher|nbconvert|p3_test|p1_b_diag|nerfstudio|ns-train') }
            $names = ($jobs | ForEach-Object { $_.ProcessId }) -join ","
            $jobs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            "$(Get-Date -Format 'HH:mm:ss') TRIPPED: commit headroom {0:N2} GB; stopped python jobs: $names" -f $free | Out-File $log -Append -Encoding utf8
            Start-Sleep -Seconds 20
        }
    } catch {}
    Start-Sleep -Milliseconds 1500
}
