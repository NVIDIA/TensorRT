#!/bin/bash

docker build --build-arg myuid=$(id -u) --build-arg mygid=$(id -g) --rm -t sample-bert  .
