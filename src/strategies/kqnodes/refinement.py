# refinement.py

import traceback
import numpy as np
from typing import List, Tuple
from src.constants.base import ACTUAL, EFFECT
from src.strategies.kqnodes.evaluator import evaluate_k_partition, bloques_son_validos

def refinar_busqueda_local(
    analizador, blocks: List[List[Tuple[int, int]]], k: int
) -> Tuple[List[List[Tuple[int, int]]], float, np.ndarray]:
    """
    Realiza un refinamiento local usando Búsqueda Local de Vecindario Variable (VND).
    Prueba movimientos de un solo elemento y luego intercambios (swaps) entre bloques
    para minimizar la pérdida EMD.
    
    El vecindario de swaps solo se activa si N < 20 para evitar sobrecarga combinatoria.
    El total de iteraciones de refinamiento está estrictamente limitado a 20 pasos.
    """
    try:
        best_blocks = [list(b) for b in blocks]
        best_loss, best_dist = evaluate_k_partition(analizador, best_blocks)

        n_nodos = len(analizador.sia_subsistema.indices_ncubos)
        max_pasos = 5 if n_nodos >= 20 else 20
        permite_swaps = n_nodos < 20

        mejorado = True
        paso = 0

        while mejorado and paso < max_pasos:
            mejorado = False
            paso += 1

            # ================= VECINDARIO 1: Movimientos Simples =================
            for bloque_origen_idx in range(k):
                bloque_origen = best_blocks[bloque_origen_idx]
                if len(bloque_origen) <= 1:
                    continue

                for elem in list(bloque_origen):
                    # Validar si remover el elemento deja el bloque origen sin variables EFFECT
                    if elem[0] == EFFECT:
                        tiene_otro_effect = any(
                            t == EFFECT and not (t == elem[0] and val == elem[1])
                            for t, val in bloque_origen
                        )
                        if not tiene_otro_effect:
                            continue

                    for bloque_destino_idx in range(k):
                        if bloque_destino_idx == bloque_origen_idx:
                            continue

                        # Crear copia temporal y mover
                        temp_blocks = [list(b) for b in best_blocks]
                        for x in temp_blocks[bloque_origen_idx]:
                            if x[0] == elem[0] and x[1] == elem[1]:
                                temp_blocks[bloque_origen_idx].remove(x)
                                break
                        temp_blocks[bloque_destino_idx].append(elem)

                        loss, dist = evaluate_k_partition(analizador, temp_blocks)
                        if loss < best_loss - 1e-7:
                            best_loss = loss
                            best_dist = dist
                            best_blocks = temp_blocks
                            mejorado = True
                            break
                    if mejorado:
                        break
                if mejorado:
                    continue

            # ================= VECINDARIO 2: Intercambios (Swaps) =================
            # Solo se activa si permitimos swaps (N < 20) y Neighborhood 1 no mejoró
            if permite_swaps and not mejorado:
                for b1_idx in range(k):
                    for b2_idx in range(b1_idx + 1, k):
                        bloque1 = best_blocks[b1_idx]
                        bloque2 = best_blocks[b2_idx]

                        for elem1 in list(bloque1):
                            for elem2 in list(bloque2):
                                # Crear copia temporal
                                temp_blocks = [list(b) for b in best_blocks]
                                
                                # Remover elem1 de b1 y elem2 de b2
                                for x in temp_blocks[b1_idx]:
                                    if x[0] == elem1[0] and x[1] == elem1[1]:
                                        temp_blocks[b1_idx].remove(x)
                                        break
                                for x in temp_blocks[b2_idx]:
                                    if x[0] == elem2[0] and x[1] == elem2[1]:
                                        temp_blocks[b2_idx].remove(x)
                                        break
                                        
                                # Insertar cruzados
                                temp_blocks[b1_idx].append(elem2)
                                temp_blocks[b2_idx].append(elem1)

                                # Verificar que sigan siendo particiones válidas
                                if not bloques_son_validos(temp_blocks, k):
                                    continue

                                loss, dist = evaluate_k_partition(analizador, temp_blocks)
                                if loss < best_loss - 1e-7:
                                    best_loss = loss
                                    best_dist = dist
                                    best_blocks = temp_blocks
                                    mejorado = True
                                    break
                            if mejorado:
                                break
                        if mejorado:
                            break

        return best_blocks, best_loss, best_dist
    except Exception as e:
        traceback.print_exc()
        raise e
