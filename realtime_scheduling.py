"""Raise only the control worker one scheduling level on Windows.

No realtime priority, process-wide power change, or system configuration change.
"""
import os


def prioritize_control_thread():
    if os.name!='nt':return {'applied':False,'reason':'Windows-only scheduling hint'}
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentThread.restype=wintypes.HANDLE
    kernel.SetThreadPriority.argtypes=[wintypes.HANDLE,ctypes.c_int]
    kernel.SetThreadPriority.restype=wintypes.BOOL
    ok=kernel.SetThreadPriority(kernel.GetCurrentThread(),1)  # THREAD_PRIORITY_ABOVE_NORMAL
    return {'applied':bool(ok),'priority':'above normal','error':None if ok else ctypes.get_last_error()}
