PY3DOTVERSION=$(shell python -c "import sys; print(sys.version_info[0],sys.version_info[1],sep='.')")
CXX=g++
CXXFLAGS=-O3 -fopenmp -std=gnu++17 -Wall -fPIC `pkg-config --cflags --libs graph-tool-py$(PY3DOTVERSION)` -shared
ALL: libkcore.so

libkcore.so: kcore.hh kcore.cc
	${CXX} ${CXXFLAGS} kcore.cc -o libkcore.so
