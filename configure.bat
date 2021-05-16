@echo off
set /p cuda_version="Please specify your CUDA version("cpu" for no CUDA): "
echo %cuda_version% > config
echo Configuration completed.
echo Run `pip install .` to install graph4nlp in your environment.