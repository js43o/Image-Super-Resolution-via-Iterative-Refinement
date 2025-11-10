#!/bin/bash

mkdir -p ./dataset/e2f_part4/sr
mkdir -p ./dataset/e2f_part4/hr

for i in $(seq 6000 7999); do
  cp "./dataset/temp/fake/${i}.png" ./dataset/e2f_part4/sr/
  cp "./dataset/temp/real/${i}.png" ./dataset/e2f_part4/hr/
done
