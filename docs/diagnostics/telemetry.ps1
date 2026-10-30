$sp = "<scratch>"
$out = "$sp\telemetry.csv"
if (-not (Test-Path $out)) { "time,gpu_temp_c,gpu_power_w,gpu_mem_mib,gpu_util_pct,ram_free_gb,commit_free_gb,ac_power" | Out-File $out -Encoding ascii }
$end = (Get-Date).AddHours(10)
while ((Get-Date) -lt $end) {
    try {
        $g = (nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits) -replace ' ', ''
        $os = Get-CimInstance Win32_OperatingSystem
        $bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
        $ac = if ($bat -and $bat.BatteryStatus -eq 1) { 0 } else { 1 }
        "{0},{1},{2:N1},{3:N1},{4}" -f (Get-Date -Format "HH:mm:ss"), $g, ($os.FreePhysicalMemory/1MB), ($os.FreeVirtualMemory/1MB), $ac | Out-File $out -Append -Encoding ascii
    } catch {}
    Start-Sleep -Seconds 5
}
