#!/bin/bash

set -e  # Exit script on any error

echo "🟢 Updating the system..."
sudo apt update -y && sudo apt upgrade -y

############## System Dependencies ##############
echo "🛠️ Installing system dependencies..."
sudo apt install -y python3-venv python3-pip build-essential gcc g++ libatlas-base-dev

############## Create virtual environment ##############
if [ ! -d ".venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

############## Activate virtual environment ##############
echo "Activating the virtual environment..."
source .venv/bin/activate

############## Upgrade pip ##############
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

############## Install packages ##############
echo "Installing required packages..."
pip install --no-cache-dir \
    numpy\
    matplotlib pandas \
    ipykernel tqdm seaborn optuna jupyter \
    argparse imageio torch torchvision torchmetrics \
    scikit-learn einops reformer_pytorch optunahub cmaes PyWavelets\
    yfinance vnstock

############## Save environment ##############
echo "Exporting installed packages to requirements.txt..."
pip freeze > requirements.txt

############## Jupyter kernel setup ##############
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

############## Bash alias utility ##############
echo "Adding alias for quick env activation..."
if ! grep -q "alias activate_env=" ~/.bashrc; then
    echo "alias activate_env='source $(pwd)/.venv/bin/activate'" >> ~/.bashrc
fi

echo "Setup complete. Run: source ~/.bashrc"
