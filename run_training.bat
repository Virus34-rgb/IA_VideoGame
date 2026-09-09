@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
cd /d "C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame"
python mainV.py
pause