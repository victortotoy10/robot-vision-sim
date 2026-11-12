@echo off
title 4WD Smart Car - Estudio CAD 3D y Simulador
echo ========================================================
echo   ESTUDIO CAD 3D Y SIMULADOR 4WD SMART CAR
echo   Iniciando servidor local en http://localhost:8000
echo ========================================================
echo.
start http://localhost:8000
python -m http.server 8000
pause
