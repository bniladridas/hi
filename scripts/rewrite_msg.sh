#!/bin/bash

# Rewrite commit messages: lowercase first line
git filter-branch -f --msg-filter '
read msg
first_line=$(echo "$msg" | head -1 | tr "[:upper:]" "[:lower:]")
rest=$(echo "$msg" | tail -n +2)
echo "$first_line
$rest"
' -- --all
