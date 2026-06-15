# candidates.py

import numpy as np
import itertools
from typing import List, Tuple
from src.constants.base import ACTUAL, EFFECT, INFTY_POS
from src.strategies.kqnodes.evaluator import evaluate_k_partition, bloques_son_validos
from src.strategies.kqnodes.tree_search import build_tree_from_fusions, find_all_frontiers
from src.strategies.kqnodes.permutation_selector import find_best_pairing

def generar_particion_balanceada(vertices: list, k: int) -> list[list]:
    """
    Genera una partición balanceada inicial de manera equilibrada (round-robin) sobre los vértices,
    garantizando que cada uno de los k bloques contenga al menos una variable en el tiempo futuro (EFFECT).

    El proceso se realiza en las siguientes fases:
    1. Distribución Inicial:
       - Reparte cíclicamente (round-robin) todos los vértices en k bloques.
    2. Corrección de Bloques:
       - Si algún bloque carece de variables de tipo EFFECT (tiempo futuro), busca otro bloque que tenga
         al menos dos variables de tipo EFFECT y transfiere (roba) una de ellas al bloque desbalanceado.

    Args:
        vertices (list): Lista de tuplas (tiempo, índice) que representan las variables activas del sistema.
        k (int): Número exacto de bloques requeridos para la partición.

    Returns:
        list[list]: Lista de k bloques balanceados, conteniendo cada uno variables representadas por tuplas.
    """
    bloques = [[] for _ in range(k)]
    for i, v in enumerate(vertices):
        bloques[i % k].append(v)

    for i, bloque in enumerate(bloques):
        if not any(t == EFFECT for t, _ in bloque):
            for otro in bloques:
                effects = [(t, idx) for t, idx in otro if t == EFFECT]
                if len(effects) >= 2:
                    v_robar = effects[-1]
                    otro.remove(v_robar)
                    bloque.append(v_robar)
                    break
    return bloques

def q_particiones_arbol(analizador, indices: list, k: int, max_limit: int = 8) -> list[list[list[int]]]:
    """
    Genera un conjunto de particiones candidatas utilizando el árbol de fusiones del algoritmo Q.

    El proceso se realiza en las siguientes fases:
    1. Respaldo y Limpieza:
       - Guarda el historial de fusiones y la memoria del analizador para no interferir con el estado principal.
       - Limpia las estructuras temporales de fusión en el analizador.
    2. Ejecución del Algoritmo Q:
       - Ejecuta la estrategia Q sobre las variables presentes para registrar la jerarquía de fusiones.
    3. Construcción del Árbol y Búsqueda de Fronteras:
       - Construye el árbol jerárquico a partir de la secuencia de fusiones.
       - Encuentra todas las fronteras válidas de tamaño k en el árbol (particiones contiguas en el árbol).
    4. Filtrado y Ordenamiento:
       - Convierte los bloques de frontera a listas de índices.
       - Ordena las particiones según su balance (minimizando la varianza del tamaño de los bloques).
       - Retorna una lista limitada por max_limit para mantener la complejidad acotada.

    Args:
        analizador: Instancia de la estrategia que realiza la ejecución.
        indices (list): Lista de índices de variables sobre las cuales aplicar el algoritmo Q.
        k (int): Número de bloques requeridos en la partición.
        max_limit (int, opcional): Límite máximo de particiones a retornar. Por defecto es 8.

    Returns:
        list[list[list[int]]]: Lista de particiones, donde cada partición es una lista de k bloques de índices.
    """
    if len(indices) < k:
        return []

    vertices = [(ACTUAL, idx) for idx in indices]

    hist_backup = analizador.fusion_history.copy()
    mem_backup = dict(analizador.memoria_grupo_candidato)
    analizador.fusion_history.clear()
    analizador.memoria_grupo_candidato.clear()

    try:
        analizador.algorithm(vertices)
        arbol = build_tree_from_fusions(analizador.fusion_history)
        if arbol is None:
            return []

        frontiers = find_all_frontiers(arbol, k)
        particiones = []
        for frontier in frontiers:
            grupos = []
            for block in frontier:
                idxs = sorted([idx for _, idx in block])
                grupos.append(idxs)
            particiones.append(grupos)
            
        # Ordenar por simplicidad o balance y truncar a max_limit para control de complejidad
        particiones.sort(key=lambda p: sum(abs(len(b) - len(indices)/k) for b in p))
        return particiones[:max_limit]
    except Exception as e:
        print(f"[q_particiones_arbol] Error: {e}")
        return []
    finally:
        analizador.fusion_history = hist_backup
        analizador.memoria_grupo_candidato = mem_backup

def particiones_estructurales(analizador, indices: list, k: int, max_limit: int = 5) -> list[list[list[int]]]:
    """
    Genera particiones candidatas utilizando heurísticas estructurales del sistema (bloques contiguos,
    distribuciones cíclicas y agrupaciones basadas en la magnitud de probabilidad marginal).

    El proceso se realiza en las siguientes fases:
    1. Particiones por Bloques Contiguos:
       - Realiza rotaciones del conjunto de índices y los divide en bloques adyacentes de tamaño homogéneo.
    2. Particiones Cíclicas (Round-Robin):
       - Distribuye cíclicamente los índices rotados entre los k bloques.
    3. Agrupación por Marginales:
       - Ordena los índices de acuerdo con el valor de su probabilidad marginal en el subsistema.
       - Distribuye cíclicamente para agrupar variables de magnitudes similares.
    4. Filtrado y Unificación:
       - Agrega las particiones válidas sin duplicar y respetando el límite max_limit.

    Args:
        analizador: Instancia de la estrategia que contiene las distribuciones marginales.
        indices (list): Lista de índices de variables a particionar.
        k (int): Número de bloques requeridos en la partición.
        max_limit (int, opcional): Límite máximo de particiones a retornar. Por defecto es 5.

    Returns:
        list[list[list[int]]]: Lista de particiones estructurales, cada una compuesta por k bloques de índices.
    """
    n = len(indices)
    particiones = []
    seen = set()

    def agregar(grupos):
        if len(grupos) != k or any(len(g) == 0 for g in grupos):
            return
        sig = tuple(sorted(tuple(g) for g in grupos))
        if sig not in seen:
            seen.add(sig)
            particiones.append([list(g) for g in grupos])

    tam = max(1, n // k)
    for offset in range(k):
        rotado = indices[offset:] + indices[:offset]
        grupos = [sorted(rotado[i * tam : (i + 1) * tam]) for i in range(k)]
        sobrantes = rotado[k * tam :]
        for i, idx in enumerate(sobrantes):
            grupos[i % k].append(idx)
            grupos[i % k].sort()
        agregar(grupos)

    for offset in range(min(k * 3, n)):
        rotado = indices[offset:] + indices[:offset]
        grupos = [[] for _ in range(k)]
        for i, idx in enumerate(rotado):
            grupos[i % k].append(idx)
        agregar(grupos)

    n_dist = len(analizador.sia_dists_marginales)
    if indices and max(indices, default=0) < n_dist:
        indices_por_dist = sorted(
            indices, key=lambda i: analizador.sia_dists_marginales[i]
        )
        grupos = [[] for _ in range(k)]
        for i, idx in enumerate(indices_por_dist):
            grupos[i % k].append(idx)
        agregar(grupos)

    return particiones[:max_limit]

def generar_pool_candidatos(analizador, presente, futuro, k):
    """
    Construye el pool final de candidatos cruzando y permutando las particiones de alcance futuro y mecanismo presente,
    adaptando la profundidad de la búsqueda dinámicamente según el tamaño del sistema N para evitar OOM y optimizar el tiempo.

    El proceso se realiza en las siguientes fases:
    1. Adaptación de Complejidad Dinámica:
       - Si N > 22 (sistema extremadamente grande), se activa el bypass del árbol Q, limitando la búsqueda a 1 candidato estructural para garantizar ejecución inmediata.
       - Para tamaños menores, escala los límites de candidatos de árbol, candidatos estructurales, parejas cruzadas a evaluar (max_pairs) y el tamaño del pool final (tope).
    2. Generación de Particiones Unilaterales:
       - Genera particiones del árbol Q y estructurales tanto para el alcance futuro como para el mecanismo presente.
    3. Emparejamiento Óptimo (Cruzamiento):
       - Para cada combinación de partición de alcance y partición de mecanismo, encuentra la mejor asociación o asignación (permutación) de bloques mediante la función de emparejamiento húngaro/permutación optimizada.
    4. Depuración y Ordenamiento:
       - Elimina particiones duplicadas, ordena las candidatas finales por su pérdida EMD calculada y retorna las mejores clasificadas dentro del límite (tope).

    Args:
        analizador: Instancia de la estrategia que almacena el estado del subsistema.
        presente: Lista de variables del tiempo presente (ACTUAL).
        futuro: Lista de variables del tiempo futuro (EFFECT).
        k (int): Cantidad de bloques de la partición.

    Returns:
        list: Lista de las mejores k-particiones completas candidatas (estructuras de bloques con tuplas).
    """
    indices_futuros = sorted(set(idx for _, idx in futuro))
    indices_presentes = sorted(set(idx for _, idx in presente))
    n = len(indices_futuros)

    # --- Adaptador de Complejidad Dinámico ---
    if n > 22:
        # Bypass Q-tree y limitar a exactamente 1 candidato para ejecución en ~20 segundos
        parts_futuras = []
        parts_presentes = []
        parts_struct_f = particiones_estructurales(analizador, indices_futuros, k, max_limit=1)
        parts_struct_p = particiones_estructurales(analizador, indices_presentes, k, max_limit=1)
        max_pairs = 1
        tope = 1
        print(f"   [Bypass Q-Tree] N={n} > 22: limitando a 1 candidato estructural para ejecución ultra rápida.")
    else:
        if n >= 20:
            top_q = 2
            top_struct = 2
            max_pairs = 4
            tope = 3
        elif n >= 15:
            top_q = 4
            top_struct = 3
            max_pairs = 12
            tope = 30
        else:
            top_q = 8
            top_struct = 5
            max_pairs = 30
            tope = 50

        # 1. Generar particiones de alcance futuro y mecanismo presente
        parts_futuras = q_particiones_arbol(analizador, indices_futuros, k, max_limit=top_q)
        parts_presentes = q_particiones_arbol(analizador, indices_presentes, k, max_limit=top_q)
        parts_struct_f = particiones_estructurales(analizador, indices_futuros, k, max_limit=top_struct)
        parts_struct_p = particiones_estructurales(analizador, indices_presentes, k, max_limit=top_struct)

    all_futuros = parts_futuras + parts_struct_f
    all_presentes = parts_presentes + parts_struct_p

    # Remover duplicados preservando orden
    unique_futuros = []
    seen_f = set()
    for p in all_futuros:
        sig = tuple(sorted(tuple(sorted(block)) for block in p))
        if sig not in seen_f:
            seen_f.add(sig)
            unique_futuros.append(p)

    unique_presentes = []
    seen_p = set()
    for p in all_presentes:
        sig = tuple(sorted(tuple(sorted(block)) for block in p))
        if sig not in seen_p:
            seen_p.add(sig)
            unique_presentes.append(p)

    pool_candidates = []
    seen_candidate_sig = set()
    
    # 2. Cruzar particiones limitando por max_pairs para evitar sobrecarga combinatoria
    pairs_evaluated = 0
    
    for p_alc in unique_futuros:
        for p_mec in unique_presentes:
            if pairs_evaluated >= max_pairs:
                break
                
            best_blocks, best_loss = find_best_pairing(analizador, p_alc, p_mec, k)
            if best_blocks is not None:
                sig = tuple(sorted(tuple(sorted(b)) for b in best_blocks))
                if sig not in seen_candidate_sig:
                    seen_candidate_sig.add(sig)
                    pool_candidates.append((best_blocks, best_loss))
            
            pairs_evaluated += 1
        if pairs_evaluated >= max_pairs:
            break

    # 3. Ordenar los candidatos por pérdida antes de truncar
    pool_candidates.sort(key=lambda x: x[1])
    sorted_pool = [blocks for blocks, loss in pool_candidates]

    print(f"→ Candidatos totales en pool pre-ordenados: {len(sorted_pool)}")

    # 4. Fallback si el pool está vacío
    if not sorted_pool:
        grupos_f = [indices_futuros[i::k] for i in range(k)]
        grupos_p = [indices_presentes[i::k] for i in range(k)]
        best_blocks, _ = find_best_pairing(analizador, grupos_f, grupos_p, k)
        if best_blocks:
            sorted_pool.append(best_blocks)

    return sorted_pool[:tope]

def ejecutar_busqueda_exhaustiva(
    analizador, k: int
) -> Tuple[List[List[Tuple[int, int]]], float, np.ndarray]:
    """
    Genera y evalúa exhaustivamente el espacio completo de k-particiones posibles para sistemas pequeños,
    asegurando encontrar la partición con la menor pérdida global de información.

    El proceso se realiza en las siguientes fases:
    1. Generación de Particiones de Alcance:
       - Utiliza fuerza bruta para generar todas las particiones posibles de tamaño k del conjunto de variables futuras (alcance).
    2. Combinación y Permutación con Mecanismos:
       - Para cada partición de alcance, recorre todas las asignaciones posibles de variables presentes (mecanismo) a los k bloques.
    3. Evaluación Global:
       - Evalúa la k-partición generada llamando a evaluate_k_partition.
       - Almacena y actualiza el menor valor de pérdida EMD encontrado junto con sus bloques y distribución marginal.

    Args:
        analizador: Instancia de la estrategia que realiza el proceso y evalúa las distancias.
        k (int): Número de bloques requerido en la partición.

    Returns:
        Tuple[List[List[Tuple[int, int]]], float, np.ndarray]: Tupla con la mejor configuración de bloques de tamaño k,
        su pérdida asociada, y el vector de distribución de probabilidad correspondiente.
    """
    from src.funcs.force import generar_k_particiones_conjunto
    
    indices_futuros = sorted(list(analizador.sia_subsistema.indices_ncubos))
    indices_presentes = sorted(list(analizador.sia_subsistema.dims_ncubos))
    
    parts_futuras = list(generar_k_particiones_conjunto(indices_futuros, k))
    
    best_loss = INFTY_POS
    best_blocks = None
    best_dist = None
    
    n_pres = len(indices_presentes)
    
    for p_alc in parts_futuras:
        for asignacion in itertools.product(range(k), repeat=n_pres):
            p_mec = [[] for _ in range(k)]
            for val_idx, block_idx in zip(indices_presentes, asignacion):
                p_mec[block_idx].append(val_idx)
            
            blocks = []
            for alc, mec in zip(p_alc, p_mec):
                bloque = [(EFFECT, idx) for idx in alc] + [(ACTUAL, idx) for idx in mec]
                blocks.append(bloque)
            
            loss, dist = evaluate_k_partition(analizador, blocks)
            if loss < best_loss:
                best_loss = loss
                best_blocks = blocks
                best_dist = dist
                
    return best_blocks, best_loss, best_dist
