#!/bin/bash

for value in {0.0,0.001,0.005,0.01}
do
    python runner.py --aug_reg=$value
done
echo All done
