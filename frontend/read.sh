#!/bin/bash

# List of directories to exclude
EXCLUDE_DIRS="(.vscode|.mvn|target|node_modules|public|assets)"

# List of file names to exclude
EXCLUDE_FILES="(.gitignore|package-lock.json|mvnw|mvnw.cmd|output.txt|read.sh)"

# File extensions to exclude
EXCLUDE_EXTENSIONS="(.png|.jpg)"

# Function to print file content
print_file_content() {
    local file=$1
    echo "File: $file"
    echo "-------------------------"
    cat "$file"
    echo
}

# Traverse the current directory recursively
find . -type f | while read -r file; do
    # Skip files in excluded directories
    if [[ $file =~ $EXCLUDE_DIRS ]]; then
        continue
    fi

    # Skip excluded files
    base_file=$(basename "$file")
    if [[ $base_file =~ $EXCLUDE_FILES ]]; then
        continue
    fi

    # Skip files with excluded extensions
    if [[ $base_file =~ $EXCLUDE_EXTENSIONS ]]; then
        continue
    fi

    # Print file path and content
    print_file_content "$file"
done
