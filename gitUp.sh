#!/bin/bash

datetime=$(date '+%Y-%m-%d %H:%M:%S')
commit_msg="[$datetime] Newest Version Updated"

# Git commands
git add -A
git commit -m "$commit_msg"
git push origin main
