#!/usr/bin/env bash
echo "checking root dir"
pycodestyle *.py

echo "checking src dir"
pycodestyle src

echo "checking scripts dir"
pycodestyle scripts

echo "checking datasets dir"
pycodestyle datasets

echo "done"
