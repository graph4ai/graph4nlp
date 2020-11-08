# Documentation Development Guide

## Building Static Documentation Webpages

Sphinx provides convenient building configuration. If you make some changes to the documentation source file and want 
to rebuild the whole documentation. Just run `makedoc.sh`.

## Preview documentation locally

To view the documentations, you need to run a http server at `build/html`. Suppose you are in `graph4nlp/docs` now,
please run the following commands to start a simple http server using python:
```
cd build/html
python -m http.server 80
```
You can replace the port 80 by other port number if port 80 is occupied.

Then you can go to the following [link](http://localhost:80) in your web browser to view the doc.