@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;C:\Program Files\CMake\bin;C:\Users\verma\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set TORCH_CUDA_ARCH_LIST=12.0
set DISTUTILS_USE_SDK=1
set "NVCC_APPEND_FLAGS=-DWIN32_LEAN_AND_MEAN -DNOMINMAX"
set PYTHONIOENCODING=utf-8
%*
