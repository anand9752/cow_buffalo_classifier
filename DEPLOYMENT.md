# 🚀 Deployment Guide for Cattle & Breed Classifier

This guide provides step-by-step instructions for deploying the AI-Powered Cattle & Breed Classifier to various cloud platforms.

## 📋 Table of Contents
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Heroku Deployment](#heroku-deployment)
- [Docker Deployment](#docker-deployment)
- [Local Development Setup](#local-development-setup)
- [Troubleshooting](#troubleshooting)

## 🌟 Streamlit Cloud Deployment (Recommended)

Streamlit Cloud is the easiest and most cost-effective way to deploy your Streamlit app.

### Prerequisites
- GitHub account
- Git installed on your local machine
- Your application code ready

### Step 1: Prepare Your Repository

1. **Initialize Git Repository** (if not already done):
   ```bash
   cd cow_buffalo_classifier
   git init
   git add .
   git commit -m "Initial commit - Cattle & Breed Classifier"
   ```

2. **Create GitHub Repository**:
   - Go to [GitHub](https://github.com)
   - Click "New repository"
   - Name it `cattle-breed-classifier`
   - Make it public (required for free Streamlit Cloud)
   - Don't initialize with README (you already have files)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/cattle-breed-classifier.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy Your App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/cattle-breed-classifier`
   - Set main file path: `cattle_with_breed_classifier.py`
   - Choose branch: `main`
   - Click "Deploy!"

3. **Wait for Deployment**:
   - Streamlit Cloud will install dependencies from `requirements.txt`
   - First deployment may take 5-10 minutes
   - You'll get a unique URL like `https://your-app-name.streamlit.app`

### Step 3: Configuration (Optional)

If you need to add secrets or environment variables:
1. Go to your app's dashboard on Streamlit Cloud
2. Click "Settings" → "Secrets"
3. Add any sensitive configuration in TOML format

## 🔧 Heroku Deployment

For more control and custom domains, you can deploy to Heroku.

### Prerequisites
- Heroku account
- Heroku CLI installed

### Step 1: Prepare Heroku Files

1. **Create Procfile**:
   ```bash
   echo "web: streamlit run cattle_with_breed_classifier.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   ```

2. **Create setup.sh**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[general]
   email = \"your-email@domain.com\"
   " > ~/.streamlit/credentials.toml
   echo "[server]
   headless = true
   enableCORS=false
   port = $PORT
   " > ~/.streamlit/config.toml
   ```

### Step 2: Deploy to Heroku

1. **Login to Heroku**:
   ```bash
   heroku login
   ```

2. **Create Heroku App**:
   ```bash
   heroku create your-cattle-classifier
   ```

3. **Deploy**:
   ```bash
   git add .
   git commit -m "Prepare for Heroku deployment"
   git push heroku main
   ```

## 🐳 Docker Deployment

For containerized deployment across different platforms.

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application files
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "cattle_with_breed_classifier.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Run

1. **Build Docker Image**:
   ```bash
   docker build -t cattle-classifier .
   ```

2. **Run Container**:
   ```bash
   docker run -p 8501:8501 cattle-classifier
   ```

### Step 3: Deploy to Cloud

You can deploy the Docker container to:
- **AWS ECS**
- **Google Cloud Run**
- **Azure Container Instances**
- **DigitalOcean App Platform**

## 💻 Local Development Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/cattle-breed-classifier.git
cd cattle-breed-classifier
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Application
```bash
streamlit run cattle_with_breed_classifier.py
```

## 🔧 Environment Variables and Configuration

### Required Files for Deployment:
- ✅ `requirements.txt` - Python dependencies
- ✅ `cattle_with_breed_classifier.py` - Main application
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ `models/` directory with model files

### Optional Files:
- `Procfile` - For Heroku deployment
- `Dockerfile` - For Docker deployment
- `.streamlit/secrets.toml` - For sensitive data (don't commit!)

## 🛠️ Troubleshooting

### Common Issues and Solutions:

#### 1. **Import Errors**
```
ModuleNotFoundError: No module named 'torch'
```
**Solution**: Ensure all dependencies are in `requirements.txt` with correct versions.

#### 2. **Memory Issues**
```
Killed due to memory usage
```
**Solution**: 
- Use CPU versions of PyTorch: `torch` instead of `torch+cpu`
- Optimize model loading in your code
- Consider upgrading to paid hosting tiers

#### 3. **File Not Found Errors**
```
FileNotFoundError: models/best_cow_buffalo_none_classifier.pth
```
**Solution**: 
- Ensure model files are included in your repository
- Check file paths are correct and case-sensitive
- Verify model files aren't in `.gitignore`

#### 4. **Port Issues (Local)**
```
Port 8501 is already in use
```
**Solution**:
```bash
streamlit run cattle_with_breed_classifier.py --server.port 8502
```

#### 5. **Streamlit Cloud Build Failures**
- Check `requirements.txt` for incompatible versions
- Ensure Python version compatibility
- Look at build logs for specific error messages

### Performance Optimization Tips:

1. **Model Loading**:
   ```python
   @st.cache_resource
   def load_model():
       # Your model loading code
       return model
   ```

2. **Data Caching**:
   ```python
   @st.cache_data
   def load_breed_database():
       # Your data loading code
       return data
   ```

3. **Image Processing**:
   - Resize images before processing
   - Use appropriate image formats (JPEG for photos)
   - Implement image compression

## 🔒 Security Considerations

1. **Secrets Management**:
   - Never commit API keys or passwords
   - Use Streamlit Cloud secrets for sensitive data
   - Use environment variables for configuration

2. **File Upload Security**:
   - Validate file types and sizes
   - Scan uploaded files for malware (if applicable)
   - Implement rate limiting

3. **Model Security**:
   - Ensure model files are from trusted sources
   - Consider model encryption for sensitive applications

## 📞 Support and Resources

### Official Documentation:
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [PyTorch Documentation](https://pytorch.org/docs)

### Community Support:
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [GitHub Issues](https://github.com/YOUR_USERNAME/cattle-breed-classifier/issues)

### Monitoring and Analytics:
- Monitor app performance through Streamlit Cloud dashboard
- Use Google Analytics for usage tracking
- Implement error logging for debugging

---

## 🎉 Congratulations!

Your Cattle & Breed Classifier is now ready for deployment! Choose the deployment method that best fits your needs and follow the step-by-step instructions above.

For any issues or questions, please refer to the troubleshooting section or create an issue in the GitHub repository.