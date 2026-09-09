@echo off
echo ======================================
echo Inicializando entorno de compilacion...
echo ======================================
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if %errorlevel% neq 0 (
    echo ERROR: No se pudo cargar el entorno de compilacion.
    pause
    exit /b 1
)

echo.
echo ======================================
echo Cambiando al directorio del proyecto...
echo ======================================
cd /d "C:\Users\mende\Universidad\AI-Adaptive-Game\CastleGame"
if %errorlevel% neq 0 (
    echo ERROR: No se pudo cambiar al directorio.
    pause
    exit /b 1
)
echo Directorio actual: %cd%

echo.
echo ======================================
echo Ejecutando test de equivalencia...
echo ======================================
python -m tests.test_torch_compile_equivalence
if %errorlevel% neq 0 (
    echo ERROR: El test fallo con codigo %errorlevel%.
    pause
    exit /b 1
)

echo.
echo ======================================
echo Test completado con exito.
echo ======================================
pause