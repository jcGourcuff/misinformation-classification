#!/bin/bash

where=$1
run_pylint=true

if [ -z "$where" ]; then
    where="."
    run_pylint=false
fi

isort $where
black $where
mypy $where

if $run_pylint; then 
    pylint $where 
fi