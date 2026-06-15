# permutation_selector.py

import itertools
import numpy as np
from typing import List, Tuple
from src.constants.base import ACTUAL, EFFECT, INFTY_POS
from src.strategies.kqnodes.evaluator import bipartir_y_emd
from src.funcs.iit import emd_efecto

def find_best_pairing(analizador, part_alcance: List[List[int]], part_mecanismo: List[List[int]], k: int) -> Tuple[List[List[Tuple[int, int]]], float]:
    """
    Evalúa todas las k! permutaciones de emparejamiento entre bloques de mecanismo (presente) 
    y bloques de alcance (futuro) para encontrar la combinación que minimice la pérdida EMD.
    Para garantizar alta velocidad, utiliza la caché de bloques individuales de forma que 
    las permutaciones se evalúan instantáneamente en memoria en microsegundos sin hacer nuevos cálculos de EMD sobre el subsistema.

    El proceso se realiza en dos fases principales:

    1. Pre-cálculo e Indexación de Bloques (Fase Cache):
       - Se evalúan las k^2 combinaciones posibles de emparejamiento (alcance_i con mecanismo_j) 
         invocando la función 'bipartir_y_emd' (que a su vez utiliza la optimización Slice-First).
       - Los vectores de distribución marginal obtenidos de cada bloque se almacenan temporalmente 
         en una matriz de referencia de tamaño k x k.

    2. Búsqueda y Selección del Emparejamiento Óptimo (Fase Permutaciones):
       - Se recorren todas las k! permutaciones (de tamaño 120 para k=5) de asignación de bloques.
       - Para cada permutación, se construye la distribución producto total hat_p en memoria 
         reutilizando los vectores marginales pre-calculados, y se evalúa el costo EMD total 
         comparándolo directamente contra las marginales del subsistema.
       - Se seleccionan e instancian los bloques de variables (tiempo, índice) de la permutación que 
         produjo la menor pérdida de información combinada.

    Args:
        analizador (KQNodes): Instancia del analizador que contiene la estructura del subsistema y variables de estado.
        part_alcance (List[List[int]]): Partición del conjunto de alcance futuro del sistema en k bloques.
        part_mecanismo (List[List[int]]): Partición del conjunto de mecanismo presente del sistema en k bloques.
        k (int): Cantidad exacta de bloques permitidos para la partición.

    Returns:
        Tuple[List[List[Tuple[int, int]]], float]: Una tupla conteniendo:
            - La lista de bloques de la mejor k-partición emparejada (cada bloque contiene tuplas (tiempo, índice)).
            - El valor mínimo de pérdida EMD asociado a este emparejamiento óptimo.
    """
    indices_sistema = list(analizador.sia_subsistema.indices_ncubos)
    
    block_marginals = [[None for _ in range(k)] for _ in range(k)]
    for i in range(k):
        alc = sorted(part_alcance[i])
        for j in range(k):
            mec = sorted(part_mecanismo[j])
            _, dist = bipartir_y_emd(analizador, alc, mec, solo_emd=False)
            block_marginals[i][j] = dist

    best_loss = INFTY_POS
    best_blocks = None
    
    map_effect = {idx: (EFFECT, idx) for idx in indices_sistema}
    indices_presentes = sorted(set(idx for b in part_mecanismo for idx in b))
    map_actual = {idx: (ACTUAL, idx) for idx in indices_presentes}

    for perm in itertools.permutations(range(k)):
        hat_p = np.zeros(len(indices_sistema), dtype=np.float32)
        valid = True
        
        for i in range(k):
            dist_bloque = block_marginals[i][perm[i]]
            if dist_bloque is None:
                valid = False
                break
            alc = part_alcance[i]
            for idx in alc:
                if idx in indices_sistema:
                    pos = indices_sistema.index(idx)
                    hat_p[pos] = dist_bloque[pos]
                    
        if not valid:
            continue
            
        loss = emd_efecto(hat_p, analizador.sia_dists_marginales)
        if loss < best_loss:
            best_loss = loss
            blocks = []
            for i in range(k):
                bloque = [map_effect[idx] for idx in part_alcance[i] if idx in map_effect]
                bloque += [map_actual[idx] for idx in part_mecanismo[perm[i]] if idx in map_actual]
                blocks.append(bloque)
            best_blocks = blocks

    return best_blocks, best_loss
