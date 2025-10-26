#!/bin/bash

# Rewrite commit messages: lowercase first line and truncate to 40 chars
git filter-branch -f --msg-filter '
read msg
first_line=$(echo "$msg" | head -1 | tr "[:upper:]" "[:lower:]" | cut -c1-40)
rest=$(echo "$msg" | tail -n +2)
echo "$first_line
$rest"
' -- --all