# test_bruteforce_k3.py
import numpy as np
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.kqnodes.kqnodes import KQNodes
from src.strategies.force import BruteForce

print("Starting correctness test for k=3...")

# Seed for reproducibility
np.random.seed(42)
tpm = np.random.rand(16, 4).astype(np.float32)
# Normalize rows to form valid probabilities
for i in range(16):
    tpm[i] /= tpm[i].sum()

print("1. Initializing KQNodes...")
kq = KQNodes(tpm)

print("2. Initializing BruteForce...")
bf = BruteForce(tpm)

estado_inicial = "1000"
condicion = "1111"
alcance = "1111"
mecanismo = "1111"

print("\n--- Running KQNodes (k=3) ---")
sol_kq = kq.aplicar_estrategia_k(estado_inicial, condicion, alcance, mecanismo, k=3)
print(f"KQNodes loss: {sol_kq.perdida:.6f}")
print(f"KQNodes partition:\n{sol_kq.particion}")

print("\n--- Running BruteForce (k=3) ---")
sol_bf = bf.aplicar_estrategia_k(estado_inicial, condicion, alcance, mecanismo, k=3)
print(f"BruteForce loss: {sol_bf.perdida:.6f}")
print(f"BruteForce partition:\n{sol_bf.particion}")

assert np.allclose(sol_kq.perdida, sol_bf.perdida, atol=1e-5), f"Mismatch! KQNodes: {sol_kq.perdida}, BruteForce: {sol_bf.perdida}"
print("SUCCESS: Loss values match perfectly!")
