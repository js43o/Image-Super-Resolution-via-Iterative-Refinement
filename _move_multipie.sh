#!/bin/bash

mkdir -p ./dataset/multipie_part1/sr
mkdir -p ./dataset/multipie_part1/hr
mkdir -p ./dataset/multipie_part2/sr
mkdir -p ./dataset/multipie_part2/hr

for i in $(seq 201 225); do
  cp ./dataset/multipie/test/sr_128_128/${i}*.png ./dataset/multipie_part1/sr/
  cp ./dataset/multipie/test/hr_128/${i}*.png ./dataset/multipie_part1/hr/
done

for i in $(seq 226 249); do
  cp ./dataset/multipie/test/sr_128_128/${i}*.png ./dataset/multipie_part2/sr/
  cp ./dataset/multipie/test/hr_128/${i}*.png ./dataset/multipie_part2/hr/
done
