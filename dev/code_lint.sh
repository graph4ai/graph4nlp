#!/bin/bash -e
# Run this script at project root by "./dev/code_lint.sh" before you commit

{
  black --version | grep -E "21.4b2" > /dev/null
} || {
  echo "Linter requires 'black==21.4b2' !"
  exit 1
}

ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 4.3* ]]; then
  echo "Linter requires isort==4.3.21 !"
  exit 1
fi

echo "Running isort on file: "$1
isort -y --atomic $1

echo "Reformatting the file: "$1
black $1

echo "Conducting code linter flake8 on file: "$1
flake8 $1
