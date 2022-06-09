#!/bin/sh
dir1="./source"
while inotifywait -qqre modify "$dir1"; do
    make clean html
done