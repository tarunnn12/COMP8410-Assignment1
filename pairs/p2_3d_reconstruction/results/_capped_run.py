
import sys, subprocess, ctypes
from ctypes import wintypes
cap_gb, cmd = float(sys.argv[1]), sys.argv[2:]
class IO(ctypes.Structure): _fields_ = [(n, ctypes.c_ulonglong) for n in ("r1", "r2", "r3", "r4", "r5", "r6")]
class BASIC(ctypes.Structure): _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
    ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]
class EXT(ctypes.Structure): _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.CreateJobObjectW.restype = wintypes.HANDLE
k.GetCurrentProcess.restype = wintypes.HANDLE
k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
job = k.CreateJobObjectW(None, None); info = EXT()
info.Basic.LimitFlags = 0x200 | 0x2000          # job memory limit, kill everything when the launcher exits
info.JobMemoryLimit = int(cap_gb * 2**30)
assert k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)), ctypes.get_last_error()
assert k.AssignProcessToJobObject(job, k.GetCurrentProcess()), ctypes.get_last_error()
sys.exit(subprocess.call(cmd))
