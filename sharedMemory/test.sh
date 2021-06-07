#!/bin/bash

printf "MD5Shared 3 f3abb86bd34cf4d52698f14c0da1dc60"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Shared 3 f3abb86bd34cf4d52698f14c0da1dc60
    done;

printf "________________________________________________________________\n"

printf "MD5Shared 4 02c425157ecd32f259548b33402ff6d3"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Shared 4 02c425157ecd32f259548b33402ff6d3
    done;

printf "________________________________________________________________\n"

printf "MD5Shared 5 95ebc3c7b3b9f1d2c40fec14415d3cb8"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Shared 5 95ebc3c7b3b9f1d2c40fec14415d3cb8
    done;

printf"**************************************************************************************"

printf "MD5Private 3 f3abb86bd34cf4d52698f14c0da1dc60"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Private 3 f3abb86bd34cf4d52698f14c0da1dc60
    done;

printf "________________________________________________________________\n"

printf "MD5Private 4 02c425157ecd32f259548b33402ff6d3"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Private 4 02c425157ecd32f259548b33402ff6d3
    done;

printf "________________________________________________________________\n"


printf "MD5Private 5 95ebc3c7b3b9f1d2c40fec14415d3cb8"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Private 5 95ebc3c7b3b9f1d2c40fec14415d3cb8
    done;
