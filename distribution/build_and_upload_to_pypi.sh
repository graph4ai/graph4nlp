cd ..
rm -rf build dist
python setup.py bdist_wheel --universal
python setup.py sdist

# Current network unavailable in Mainland China
#twine upload dist/*
