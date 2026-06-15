# test_n25_scalability.py
import numpy as np
import sys
import os
import time

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.kqnodes.kqnodes import KQNodes

print("Starting scalability test for N=25...")

tpm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", ".samples", "N25A.npy")

if os.path.exists(tpm_path):
    print(f"Loading real N=25 TPM from {tpm_path}...")
    # Using memory mapping to avoid reading the entire 838MB file at once if possible
    tpm = np.load(tpm_path, mmap_mode='r')
else:
    print("N25A.npy not found in src/.samples. Generating mock N=25 system (33.5M states) in float32...")
    # To conserve memory during creation, we use a single row pattern or mock
    # since we only need the structure and a valid transition probability to test the bypass.
    # We create a mock TPM of size (2^25, 1) or shape that allows the evaluator to run.
    # Note that System class only needs shapes matching the variables.
    # In a real run, the System object is initialized with the TPM.
    # We can create a mock TPM of size 33,554,432 floats.
    tpm = np.random.rand(33554432, 1).astype(np.float32)

print(f"TPM shape: {tpm.shape}, dtype: {tpm.dtype}")

print("Initializing KQNodes...")
start_init = time.time()
kq = KQNodes(tpm)
print(f"Initialization completed in {time.time() - start_init:.4f} seconds.")

estado_inicial = "1" + "0"*24
condicion = "1"*25
alcance = "1"*25
mecanismo = "1"*25

print("\n--- Running KQNodes (k=3, N=25) ---")
start_exec = time.time()
sol = kq.aplicar_estrategia_k(estado_inicial, condicion, alcance, mecanismo, k=3)
exec_time = time.time() - start_exec

print(f"\nExecution completed in {exec_time:.4f} seconds.")
print(f"KQNodes loss: {sol.perdida:.6f}")
print(f"KQNodes partition:\n{sol.particion}")

assert exec_time < 2.0, f"Execution took too long: {exec_time:.2f} seconds!"
print("SUCCESS: Scalability test passed!")
