#!/bin/bash
# Startup script for CrewAI Data Analyst

echo "🚀 Starting CrewAI Data Analyst..."
echo "📦 Installing dependencies..."

# Install dependencies
if command -v uv &> /dev/null; then
    echo "Using uv for package management..."
    uv pip install -r requirements.txt
elif command -v pip &> /dev/null; then
    echo "Using pip for package management..."
    pip install -r requirements.txt
else
    echo "❌ Neither uv nor pip found. Please install Python package manager."
    exit 1
fi

echo "🧪 Running setup tests..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo "✅ Setup complete! Starting Streamlit app..."
    echo "🌐 Opening http://localhost:8501"
    streamlit run app.py
else
    echo "❌ Setup tests failed. Please check the error messages above."
    exit 1
fi
