@echo off
echo Please specify your CUDA version
set /p cuda_version="[1] for CUDA 9.2, [2] for CUDA 10.1, [3] for CUDA 10.2, [4] for CUDA 11.0, [5] for CUDA 11.1, [6] for CPU: "
echo %cuda_version% > config
echo Configuration completed.
echo Run `python setup.py install` to install graph4nlp in your environment.