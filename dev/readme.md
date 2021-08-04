# The develop tools

## Code Linter
This tells you how to check the code style.

### Install
```bash
pip install -r dev/requirements.txt
```

### Run Linter

Example: ``graph4nlp/pytorch/models/graph2seq.py``

**Remember: Run the following script in the root directory of the project!**
```bash
./dev/code_lint.sh graph4nlp/pytorch/models/graph2seq.py
```

1. ``isort``. It will automatically reformat the imports. It may also raise some errors.
2. ``black``. Black is a PEP 8 compliant opinionated formatter with its own style. 
3. ``flake8``. This is a code style checker. It will raise errors and warnings. Fix them all!
When you meet the error: e.g., "graph4nlp/pytorch/models/graph2seq.py:259:101: E501 line too long (111 > 100 characters)" and find it in comments, you just need to add "# noqa" at the end of the line.

Contact [me](https://github.com/AlanSwift) if you meet any problem.
