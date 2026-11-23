#!/usr/bin/env python3
"""
Fix test file to ensure all configs have proper sections and escaping
"""

import re
import os

# Read the test file
with open(os.path.join(os.path.dirname(__file__), 'test_dataset_generator.py'), 'r') as f:
    lines = f.readlines()

# Find all lines that create config['logging'] and haven't been fixed yet
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Replace any remaining single-% log formats
    if "'log_format':" in line and "'%(asctime)s" in line:
        line = line.replace("'%(asctime)s", "'%%(asctime)s")
        line = line.replace("%(levelname)s", "%%(levelname)s")
        line = line.replace("%(message)s", "%%(message)s")  
    
    output_lines.append(line)
    i += 1

# Write back
with open(os.path.join(os.path.dirname(__file__), 'test_dataset_generator.py'), 'w') as f:
    f.writelines(output_lines)

print("Fixed all log_format strings in test file")
