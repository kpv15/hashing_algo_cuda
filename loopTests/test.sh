#!/bin/bash

printf "MD5Loop 3 f3abb86bd34cf4d52698f14c0da1dc60"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Loop 3 f3abb86bd34cf4d52698f14c0da1dc60
    done;

printf "________________________________________________________________\n"

printf "MD5Loop 4 02c425157ecd32f259548b33402ff6d3"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Loop 4 02c425157ecd32f259548b33402ff6d3
    done;

printf "________________________________________________________________\n"

printf "MD5Loop 5 95ebc3c7b3b9f1d2c40fec14415d3cb8"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5Loop 5 95ebc3c7b3b9f1d2c40fec14415d3cb8
    done;

printf"#########################################################################################"

printf "MD5PragmaLoop 3 f3abb86bd34cf4d52698f14c0da1dc60"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5PragmaLoop 3 f3abb86bd34cf4d52698f14c0da1dc60
    done;

printf "________________________________________________________________\n"

printf "MD5PragmaLoop 4 02c425157ecd32f259548b33402ff6d3"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5PragmaLoop 4 02c425157ecd32f259548b33402ff6d3
    done;

printf "________________________________________________________________\n"


printf "MD5PragmaLoop 5 95ebc3c7b3b9f1d2c40fec14415d3cb8"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5PragmaLoop 5 95ebc3c7b3b9f1d2c40fec14415d3cb8
    done;

printf"#########################################################################################"

printf "MD5StepsList 3 f3abb86bd34cf4d52698f14c0da1dc60"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5StepsList 3 f3abb86bd34cf4d52698f14c0da1dc60
    done;

printf "________________________________________________________________\n"

printf "MD5StepsList 4 02c425157ecd32f259548b33402ff6d3"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5StepsList 4 02c425157ecd32f259548b33402ff6d3
    done;

printf "________________________________________________________________\n"

printf "MD5StepsList 5 95ebc3c7b3b9f1d2c40fec14415d3cb8"
for (( i = 1; i <= 10 ; i++ ))
    do
    printf "run: %d\n" $i
    ./../cmake-build-debug-remote-host/MD5StepsList 5 95ebc3c7b3b9f1d2c40fec14415d3cb8
    done;