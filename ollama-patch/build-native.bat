@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Program Files\CMake\bin;%PATH%
rem CUDA 12.3 + MSVC 14.44 workarounds
set CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set NVCC_APPEND_FLAGS=-allow-unsupported-compiler
cd /d C:\BitsPlus\VisOllamInference
cmake -S ollama -B build-ollama-full -G "Visual Studio 17 2022" -A x64 -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DCMAKE_CUDA_ARCHITECTURES=89 -DOLLAMA_BUILD_PARALLEL=16
if errorlevel 1 exit /b 1
cmake --build build-ollama-full --config Release --target ollama-llama-server-local ollama-llama-server-cuda_v12 -- /m:2
