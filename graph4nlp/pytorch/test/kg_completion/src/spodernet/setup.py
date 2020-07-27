import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "spodernet",
    version = "0.0.1",
    author = "Tim Dettmers",
    author_email = "tim.dettmers@gmail.com",
    description = ("Preprocessing framework for NLP."),
    license = "MIT",
    keywords = "deep learning NLP",
    url = "http://packages.python.org/spodernet",
    packages=['spodernet'],
    test_suite="tests",
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Deep Learning",
    ],
    install_requires=[
        'pytest',
        'numpy',
        'h5py',
        'nltk',
        'enum34',
        'spacy',
        'sklearn',
        'future'
    ]
)
