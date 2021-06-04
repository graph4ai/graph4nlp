cd ..
rm -rf build dist
./configure
python setup.py bdist_wheel --universal
python setup.py sdist

# Current network unavailable in Mainland China
twine upload dist/*
