default: html

all: html test

check: test-noninteractive test

_static/node_modules/mathjax:
	cd _static; npm install mathjax@3

html: _static/node_modules/mathjax
	sphinx-build -j auto -E -b html . build/html

update:
	sphinx-build -j auto -b html . build/html

test:
	sphinx-build -j auto -b doctest . build/test

test-noninteractive:
	sphinx-build -b doctest . build/test-ni *.rst *.md demos/inference/inference.md

clean:
	rm -rf build

