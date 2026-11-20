@echo off
REM One-click setup + run for Windows cmd. Run from the customeriq folder.
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  %PY% -m venv .venv || (echo Failed to create venv. Install Python 3.10+ & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Dependency install failed & exit /b 1)
if not exist data mkdir data
python -m src.data.sample_data --out data\sample_tourism.csv
python train_cli.py --sample
python -m pytest -q
echo.
echo Setup complete. Starting the app...
streamlit run app\streamlit_app.py
