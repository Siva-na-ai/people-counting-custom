#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "   Raspberry Pi 5 AI Camera Setup Script          "
echo "=================================================="

# Check for system dependencies
echo "Checking system requirements..."

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed."
    echo "Please install it using: sudo apt update && sudo apt install -y python3"
    exit 1
fi

# Check Git
if ! command -v git &> /dev/null; then
    echo "❌ Error: git is not installed."
    echo "Please install it using: sudo apt update && sudo apt install -y git"
    exit 1
fi

# Check if python3-venv package is installed (common issue on Raspberry Pi OS)
if ! python3 -c "import venv" &> /dev/null; then
    echo "❌ Error: python3-venv is missing."
    echo "Please run the following command to install it:"
    echo "   sudo apt update && sudo apt install -y python3-venv"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch & torchvision first
echo "Installing PyTorch & torchvision..."
pip install torch torchvision

# Install packages from requirements.txt
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Installing standard dependencies..."
    pip install numpy opencv-python torchreid
fi

# Install modlib from Sony Github repository
echo "Installing Sony modlib AITRIOS SDK..."
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git

echo "=================================================="
echo "   Setting up Qdrant Vector Database              "
echo "=================================================="
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker is not installed. You will need Docker to run Qdrant locally."
    echo "Install docker with: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
else
    echo "Docker found. You can start Qdrant by running:"
    echo "   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant"
fi

echo "=================================================="
echo "   Setup Complete!                                "
echo "=================================================="
echo "To activate the virtual environment, run:"
echo "   source venv/bin/activate"
echo ""
echo "💡 Tip: If you get OpenCV errors (e.g. libGL.so missing) when running, install:"
echo "   sudo apt update && sudo apt install -y libglib2.0-0 libgl1-mesa-glx"
echo "=================================================="
