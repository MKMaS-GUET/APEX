#!/bin/bash

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3"
cmake --build build -j16
