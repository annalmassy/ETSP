#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "Installing required packages..."
python3 -m pip install --upgrade pip || {
    echo "Error upgrading pip!"
    read -p "Press enter to exit"
    exit 1
}

echo "Installing requirements..."
python3 -m pip install -r requirements.txt || {
    echo "Error installing packages!"
    read -p "Press enter to exit"
    exit 1
}

echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')" || {
    echo "Error downloading NLTK data!"
    read -p "Press enter to exit"
    exit 1
}

echo "Starting Streamlit application..."
streamlit run app.py || {
    echo "Error starting Streamlit application!"
    read -p "Press enter to exit"
    exit 1
}

read -p "Press enter to exit"