#!/bin/bash

mkdir -p ./dataset/kface_part1/sr
mkdir -p ./dataset/kface_part1/hr

for i in $(seq 1906 1908); do
  cp "./dataset/kface/test/sr_32_128/${i}*.png" ./dataset/kface_part1/sr/
  cp "./dataset/kface/test/hr_128/${i}*.png" ./dataset/kface_part1/hr/
done

for i in $(seq 1909 1910); do
  cp "./dataset/kface/test/sr_32_128/${i}*.png" ./dataset/kface_part2/sr/
  cp "./dataset/kface/test/hr_128/${i}*.png" ./dataset/kface_part2/hr/
done
