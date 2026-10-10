@echo off
setlocal
cd /d "%~dp0"
call "D:\VS2022\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || exit /b 1
%~dp0..\cuda_tools\bin\ptxas.exe -arch=sm_86 tap.ptx -o tap.cubin || exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /W3 /MD /LD nvngx-bridge.cpp /Fe:nvngx.dll /link d3d12.lib || exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /W3 /MD /utf-8 /I"ngx" nr-lab.cpp /Fe:nr-lab.exe /link "ngx\libs\nvsdk_ngx_d.lib" d3d12.lib dxgi.lib dxguid.lib user32.lib advapi32.lib ole32.lib || exit /b 1
rem nr11: probes the D3D11 entry points of nvngx_dlssnr.dll (they are stubs: Init_Ext returns 0xBAD00001)
cl /nologo /std:c++17 /O2 /EHsc /W3 /MD /utf-8 /D_CRT_SECURE_NO_WARNINGS /I"ngx" nr11.cpp /Fe:nr11.exe /link d3d11.lib dxgi.lib user32.lib || exit /b 1
for %%D in (rt_sm86 rt_orig) do (
    copy /y nvngx.dll %%D\ >nul
    copy /y nr-lab.exe %%D\ >nul
    copy /y tap.cubin %%D\ >nul
    copy /y %~dp0..\dlss5\nvngx_dlss.dll %%D\ >nul
    copy /y %~dp0..\dlss5\nvngx_dlssg.dll %%D\ >nul
)
copy /y %~dp0..\sm86_port\nvngx_dlssnr.dll rt_sm86\ >nul
copy /y %~dp0..\dlss5\nvngx_dlssnr.dll rt_orig\ >nul
echo built
