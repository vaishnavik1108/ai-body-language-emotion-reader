@echo off
echo Starting AI Body Language Reader...
call engagement_env\Scripts\activate
streamlit run src\app.py
pause
