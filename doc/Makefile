default: html

all: html latex test

check: test-noninteractive test

html:
	sphinx-build -j auto -E -b html . build/html

update:
	sphinx-build -j auto -b html . build/html

latex:
	sphinx-build -E -b latex . build/pdf

test:
	sphinx-build -j auto -b doctest . build/test

test-noninteractive:
	sphinx-build -b doctest . build/test-ni *.rst *.md demos/inference/inference.md

clean:
	rm -rf build

