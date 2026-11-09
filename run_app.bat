@echo off
rem Lanza la app de mapa (http://localhost:8501). Doble clic para ejecutar.
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py --browser.gatherUsageStats false
pause
