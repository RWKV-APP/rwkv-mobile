"""Load the shipped library and check its C ABI, without a model download."""
import ctypes
import os
from pathlib import Path
import sys

library_path = Path(sys.argv[1]).resolve(strict=True)
# Deliberately resolve DLLs beside the artifact, not from a developer's PATH.
dll_directory = os.add_dll_directory(str(library_path.parent)) if os.name == "nt" else None
library = ctypes.CDLL(str(library_path))
backends = library.rwkvmobile_runtime_get_available_backend_names
backends.argtypes = [ctypes.c_char_p, ctypes.c_int]
backends.restype = ctypes.c_int
buffer = ctypes.create_string_buffer(4096)
assert backends(buffer, len(buffer)) == 0, "Backend enumeration failed"
names = buffer.value.decode("utf-8").strip(",").split(",")
assert "palm" in names, f"PALM missing from the packaged runtime: {names}"
print(f"Packaged library C ABI verified: {library_path.name}; backends={names}")
