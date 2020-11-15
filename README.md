# graph4nlp
Graph4nlp is the library for the easy use of Graph Neural Networks for NLP

## Packaged version for testing
Pre-released test versions have been posted to TestPyPI. Like dgl, we have different names for this library to cope with
different CUDA environments.
To enable installing using TestPyPI, first add TestPyPI index url to your pip as an extra index url:
```
pip config set global.extra-index-url https://test.pypi.org/simple/ 
```

For the CPU version, use the following command to install gaph4nlp:
```
pip install graph4nlp -U
```
For CUDA versions, try this:
```
pip install graph4nlp-cu102 -U
```
Note that the `-U` option is important since our library is rapidly iterating.

Current I only have CUDA 10.2 servers. So if you have different versions, please help me build it.
