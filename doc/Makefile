default: html

all: html test

check: test-noninteractive test

_static/MathJax:
	cd _static; \
	curl -L https://github.com/mathjax/MathJax-src/archive/refs/tags/3.2.2.tar.gz -o MathJax-3.2.2.tar.gz; \
	zcat MathJax-3.2.2.tar.gz | tar -xv; \
	ln -s MathJax-src-3.2.2 MathJax

html: _static/MathJax
	sphinx-build -j auto -E -b html . build/html

update:
	sphinx-build -j auto -b html . build/html

test:
	sphinx-build -j auto -b doctest . build/test

test-noninteractive:
	sphinx-build -b doctest . build/test-ni *.rst *.md demos/inference/inference.md

clean:
	rm -rf build

