@echo off
rem Builds bin\dxgi.dll (fallback proxy), bin\dlss5_nvngx.dll (NGX caller-identity bridge), bin\fbtest.exe.
rem Needs the DLSS SDK headers in ..\..\oss\nvidia-dlss\include (MIT, not part of this repository).
setlocal
cd /d "%~dp0"
call "D:\VS2022\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || exit /b 1
if not exist bin mkdir bin
if not exist obj mkdir obj
set NGX=..\..\oss\nvidia-dlss\include
set NVOF=..\..\oss\DLSS5-Reshade-AIO\addon\include
ml64 /nologo /c /Foobj\exports.obj exports.asm || exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /W3 /MT /D_CRT_SECURE_NO_WARNINGS /utf-8 /I"%NGX%" /I"%NVOF%" /Foobj\ /c dlss5fb.cpp || exit /b 1
link /nologo /DLL /DEF:dxgi.def /OUT:bin\dxgi.dll obj\dlss5fb.obj obj\exports.obj d3d12.lib user32.lib || exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /W3 /MT /D_CRT_SECURE_NO_WARNINGS /LD /Foobj\ ..\..\harness\nvngx-bridge.cpp /Fe:bin\dlss5_nvngx.dll /link d3d12.lib || exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /W3 /MT /D_CRT_SECURE_NO_WARNINGS /Foobj\ fbtest.cpp /Fe:bin\fbtest.exe /link d3d12.lib dxgi.lib user32.lib || exit /b 1
del bin\*.exp bin\*.lib 2>nul
echo built
