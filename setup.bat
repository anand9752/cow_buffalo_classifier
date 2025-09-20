@echo off
echo 🐄 Setting up Cattle & Breed Classifier for deployment...

REM Create .streamlit directory if it doesn't exist
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"

REM Copy configuration files
echo 📁 Setting up Streamlit configuration...
echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
echo email = "your-email@domain.com">> "%USERPROFILE%\.streamlit\credentials.toml"

echo [server]> "%USERPROFILE%\.streamlit\config.toml"
echo headless = true>> "%USERPROFILE%\.streamlit\config.toml"
echo enableCORS=false>> "%USERPROFILE%\.streamlit\config.toml"
echo port = %PORT%>> "%USERPROFILE%\.streamlit\config.toml"

echo ✅ Setup complete!
echo 🚀 You can now deploy your application!
echo.
echo 📋 Next steps:
echo 1. For Streamlit Cloud: Push to GitHub and deploy via share.streamlit.io
echo 2. For Heroku: Run 'heroku create your-app-name' and 'git push heroku main'
echo 3. For Docker: Run 'docker build -t cattle-classifier .' and 'docker run -p 8501:8501 cattle-classifier'
echo.
echo 📖 See DEPLOYMENT.md for detailed instructions!
pause