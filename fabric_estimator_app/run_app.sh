#!/bin/bash
echo "Setting up environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Streamlit App..."
streamlit run app.py
