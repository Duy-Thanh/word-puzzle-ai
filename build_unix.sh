#! /bin/bash

if command -v python3 &>/dev/null; then
    python3 --version
    pip3 install -r requirements.txt

    sudo apt-get install python3-tk
else
    echo "Python version 3 or higher must be installed before run"
fi