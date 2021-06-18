#!/bin/bash 
python setup.py bdist_wheel
pip uninstall pygo -y 
pip install dist/pygo-0.1.0-py3-none-any.whl
