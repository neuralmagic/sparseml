#!/bin/bash

if [ -n "$(which pylint3)" ]; then
    pylint3 --load-plugins=gdb,numpy -E -r n "$@" 2>&1 | egrep -v "No config file found, using default configuration" | tail -n +2
else
    echo "pylint3 not installed"
fi
