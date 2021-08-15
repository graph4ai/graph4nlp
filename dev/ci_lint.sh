echo "Running isort"
isort -c -sp .
echo "Running black"
black -l 100 --check .
echo "Running flake8"
flake8 .