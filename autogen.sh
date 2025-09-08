#!/bin/bash
# Run this to generate all the initial makefiles, etc.

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

PROJECT="graph-tool"

(test -f $srcdir/configure.ac) || {
    echo -n "**Error**: Directory \"\'$srcdir\'\" does not look like the"
    echo " top-level package directory"
    exit 1
}

aclocal -I m4 || exit $?
autoheader || exit $?
if [ `type -P libtoolize` ]; then
    libtoolize -f || exit $?
else
    glibtoolize -f || exit $?
fi
automake --add-missing --copy || exit $?
autoconf || exit $?

cd $srcdir/bundled/sparsehash
./autogen.sh
