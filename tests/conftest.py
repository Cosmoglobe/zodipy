import os

# Disable jit compilations when running tests
os.environ["NUMBA_DISABLE_JIT"] = "1"
