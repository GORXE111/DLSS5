@echo off
rem Build DLSS5Manager.exe (GUI) and dlss5.exe (CLI) for .NET Framework 4.8 with the VS2022 Roslyn compiler.
setlocal
cd /d "%~dp0"
set CSC=D:\VS2022\MSBuild\Current\Bin\Roslyn\csc.exe
if not exist "%CSC%" for /f "delims=" %%i in ('where csc 2^>nul') do set CSC=%%i
set FW=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319
set REFS=/r:"%FW%\System.Windows.Forms.dll" /r:"%FW%\System.Drawing.dll" /r:"%FW%\System.Management.dll" /r:"%FW%\System.Web.Extensions.dll" /r:"%FW%\Microsoft.CSharp.dll" /r:"%FW%\System.Core.dll"
set OPTS=/nologo /optimize+ /codepage:65001 /langversion:latest /platform:anycpu /nowarn:1702
if not exist bin mkdir bin
"%CSC%" %OPTS% /target:winexe /out:bin\DLSS5Manager.exe %REFS% Core.cs Cli.cs Gui.cs Program.cs || exit /b 1
"%CSC%" %OPTS% /target:exe /define:CLI /out:bin\dlss5.exe %REFS% Core.cs Cli.cs Gui.cs Program.cs || exit /b 1
echo built
