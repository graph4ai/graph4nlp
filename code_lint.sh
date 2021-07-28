echo "Reformatting file: "$1
black $1
echo "Reformatting imports: "$1
usort format $1
echo "Conducting code lint: "$1
flake8 $1