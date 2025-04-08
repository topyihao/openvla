#!/usr/bin/env python3

import os
import sys
import subprocess

# Save current LD_LIBRARY_PATH
old_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')

# Add the system libstdc++ path to LD_LIBRARY_PATH
# Typically it's in /usr/lib/x86_64-linux-gnu
system_lib_path = "/usr/lib/x86_64-linux-gnu"
if old_ld_library_path:
    os.environ['LD_LIBRARY_PATH'] = f"{system_lib_path}:{old_ld_library_path}"
else:
    os.environ['LD_LIBRARY_PATH'] = system_lib_path

# Execute the actual script with the modified environment
try:
    result = subprocess.run([sys.executable, "aloha_openvla.py"] + sys.argv[1:], check=True)
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    sys.exit(e.returncode)