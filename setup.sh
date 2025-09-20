#!/bin/bash

# Setup script for Cattle & Breed Classifier
# This script sets up the application for deployment

echo "🐄 Setting up Cattle & Breed Classifier for deployment..."

# Create .streamlit directory if it doesn't exist
mkdir -p ~/.streamlit/

# Copy configuration files
echo "📁 Setting up Streamlit configuration..."
echo -e '[general]\nemail = "your-email@domain.com"\n' > ~/.streamlit/credentials.toml

echo -e '[server]\nheadless = true\nenableCORS=false\nport = $PORT\n' > ~/.streamlit/config.toml

echo "✅ Setup complete!"
echo "🚀 You can now deploy your application!"

# Instructions
echo ""
echo "📋 Next steps:"
echo "1. For Streamlit Cloud: Push to GitHub and deploy via share.streamlit.io"
echo "2. For Heroku: Run 'heroku create your-app-name' and 'git push heroku main'"
echo "3. For Docker: Run 'docker build -t cattle-classifier .' and 'docker run -p 8501:8501 cattle-classifier'"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions!"