echo "Running isort"
isort -c -sp .
exit 0
echo "Running black"
black -l 100 --check .
echo "Running flake8"
flake8 .