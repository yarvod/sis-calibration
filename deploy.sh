#! /bin/bash

while getopts "t:" arg; do
  case $arg in
    t) Tag=$OPTARG;;
  esac
done

# Create tags
git commit --allow-empty -m "Release $Tag"
git tag -a $Tag -m "Version $Tag"

# Push
git push origin --tags
