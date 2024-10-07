#! /bin/bash

/usr/bin/find /Users/imran/Desktop/studies/thesis/msc_thesis/latex/fig -name "*.drawio" -exec sh -c '/Applications/draw.io.app/Contents/MacOS/draw.io --crop -x -o "${1%.drawio}.pdf" "$1"' _ {} \;
