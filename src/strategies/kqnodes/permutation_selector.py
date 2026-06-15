# permutation_selector.py

import itertools
import numpy as np
from typing import List, Tuple
from src.constants.base import ACTUAL, EFFECT, INFTY_POS
from src.strategies.kqnodes.evaluator import bipartir_y_emd
from src.funcs.iit import emd_efecto

def find_best_pairing(analizador, part_alcance: List[List[int]], part_mecanismo: List[List[int]], k: int) -> Tuple[List[List[Tuple[int, int]]], float]:
    """
    Evalúa todas las k! permutaciones de part_mecanismo emparejadas con part_alcance.
    Precalcula las k^2 distribuciones marginales usando la caché de biparticiones
    para evaluar todas las permutaciones en memoria en microsegundos.
    
    Retorna (best_blocks, best_loss).
    """
    indices_sistema = list(analizador.sia_subsistema.indices_ncubos)
    
    # 1. Precalcular y cachear las k^2 combinaciones de bloques
    # block_marginals[i][j] guarda el vector marginal para part_alcance[i] con part_mecanismo[j]
    block_marginals = [[None for _ in range(k)] for _ in range(k)]
    for i in range(k):
        alc = sorted(part_alcance[i])
        for j in range(k):
            mec = sorted(part_mecanismo[j])
            _, dist = bipartir_y_emd(analizador, alc, mec, solo_emd=False)
            block_marginals[i][j] = dist

    # 2. Evaluar todas las k! permutaciones de emparejamiento
    best_loss = INFTY_POS
    best_blocks = None
    
    map_effect = {idx: (EFFECT, idx) for idx in indices_sistema}
    indices_presentes = sorted(set(idx for b in part_mecanismo for idx in b))
    map_actual = {idx: (ACTUAL, idx) for idx in indices_presentes}

    # Iterar sobre las permutaciones del rango k
    for perm in itertools.permutations(range(k)):
        # perm[i] es el índice del bloque de mecanismo emparejado con el bloque de alcance i
        hat_p = np.zeros(len(indices_sistema), dtype=np.float32)
        valid = True
        
        for i in range(k):
            dist_bloque = block_marginals[i][perm[i]]
            if dist_bloque is None:
                valid = False
                break
            # Copiar marginales a hat_p para las variables de este bloque alcance
            alc = part_alcance[i]
            for idx in alc:
                if idx in indices_sistema:
                    pos = indices_sistema.index(idx)
                    hat_p[pos] = dist_bloque[pos]
                    
        if not valid:
            continue
            
        # Calcular la pérdida EMD de la distribución producto reconstituida
        loss = emd_efecto(hat_p, analizador.sia_dists_marginales)
        if loss < best_loss:
            best_loss = loss
            # Construir bloques finales (tiempo, índice)
            blocks = []
            for i in range(k):
                bloque = [map_effect[idx] for idx in part_alcance[i] if idx in map_effect]
                bloque += [map_actual[idx] for idx in part_mecanismo[perm[i]] if idx in map_actual]
                blocks.append(bloque)
            best_blocks = blocks

    return best_blocks, best_loss
