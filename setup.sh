#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "   Face + Body Identity Pipeline Setup Script     "
echo "=================================================="

# 1. System Requirements Check
echo "[*] Checking system requirements..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed."
    echo "Please install it using your package manager."
    exit 1
fi


# 2. Virtual Environment Setup
if ! python3 -c "import venv" &> /dev/null; then
    echo "❌ Error: python3-venv is missing."
    echo "Please install python3-venv."
    exit 1
fi

echo "[*] Creating virtual environment 'venv'..."
python3 -m venv venv

echo "[*] Activating virtual environment..."
source venv/bin/activate

# 3. Installing Python Dependencies
echo "[*] Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo "[*] Installing Core Machine Learning Dependencies..."
# Installing core dependencies for the pipeline
pip install numpy opencv-python scikit-image scikit-learn

echo "[*] Installing Qdrant Database Client..."
pip install qdrant-client

echo "[*] Installing InsightFace & ONNXRuntime..."
# Note: For GPU support, replace onnxruntime with onnxruntime-gpu
pip install onnxruntime insightface

echo "=================================================="
echo "   Setup Complete!                                "
echo "=================================================="
echo "To activate the virtual environment, run:"
echo "   source venv/bin/activate"
echo ""
echo "Then, you can start the pipeline with:"
echo "   python main.py"
echo "=================================================="
