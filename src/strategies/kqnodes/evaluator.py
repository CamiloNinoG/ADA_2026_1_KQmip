import numpy as np
from typing import List, Tuple
from src.constants.base import ACTUAL, EFFECT, INFTY_POS
from src.funcs.iit import emd_efecto

def bloques_son_validos(blocks: list, k: int) -> bool:
    """Verifica si la partición tiene exactamente k bloques y ninguno tiene alcance futuro vacío."""
    if len(blocks) != k:
        return False
    for b in blocks:
        if not any(t == EFFECT for t, _ in b):
            return False
    return True

def bipartir_y_emd(
    analizador,
    idxs_alcance: list,
    dims_mecanismo: list,
    solo_emd: bool = False,
) -> tuple[float, np.ndarray]:
    """Copia cacheada de bipartir y evaluar la EMD para el bloque actual."""
    if not hasattr(analizador, "_bipartir_emd_cache"):
        analizador._bipartir_emd_cache = {}

    clave = (tuple(sorted(idxs_alcance)), tuple(sorted(dims_mecanismo)))

    if solo_emd:
        clave_emd = ("emd_only", *clave)
        if clave_emd not in analizador._bipartir_emd_cache:
            particion = analizador.sia_subsistema.bipartir(
                np.array(idxs_alcance, dtype=np.int8),
                np.array(dims_mecanismo, dtype=np.int8),
            )
            dist = particion.distribucion_marginal()
            emd = emd_efecto(dist, analizador.sia_dists_marginales)
            analizador._bipartir_emd_cache[clave_emd] = emd
        return analizador._bipartir_emd_cache[clave_emd], None

    if clave not in analizador._bipartir_emd_cache:
        particion = analizador.sia_subsistema.bipartir(
            np.array(idxs_alcance, dtype=np.int8),
            np.array(dims_mecanismo, dtype=np.int8),
        )
        dist = particion.distribucion_marginal()
        emd = emd_efecto(dist, analizador.sia_dists_marginales)
        analizador._bipartir_emd_cache[clave] = (emd, dist)

    return analizador._bipartir_emd_cache[clave]

def evaluate_k_partition(
    analizador, blocks: List[List[Tuple[int, int]]]
) -> Tuple[float, np.ndarray]:
    """Evalúa la pérdida EMD de una k-partición dividida en bloques independientes."""
    blocks = [b for b in blocks if len(b) > 0]
    if len(blocks) < 2:
        return INFTY_POS, None

    indices_sistema = list(analizador.sia_subsistema.indices_ncubos)
    hat_p = np.zeros(len(indices_sistema), dtype=np.float32)

    for bloque in blocks:
        alc = sorted([idx for t, idx in bloque if t == EFFECT])
        mec = sorted([idx for t, idx in bloque if t == ACTUAL])

        if not alc:
            return INFTY_POS, None

        try:
            emd_bloque, dist_bloque = bipartir_y_emd(analizador, alc, mec, solo_emd=False)
        except Exception as e:
            print(f"[_evaluate_k_partition] Error bloque alc={alc} mec={mec}: {e}")
            return INFTY_POS, None

        for idx in alc:
            if idx in indices_sistema:
                pos = indices_sistema.index(idx)
                hat_p[pos] = dist_bloque[pos]

    if len(hat_p) != len(analizador.sia_dists_marginales):
        return INFTY_POS, None

    loss = emd_efecto(hat_p, analizador.sia_dists_marginales)
    return loss, hat_p

def funcion_submodular_k(analizador, deltas, omegas):
    """Calcula la ganancia de la función submodular para la iteración actual en KQNodes."""
    analizador.clave_submodular = [], []

    clave_delta_actual, clave_delta_efecto = analizador.definir_clave(deltas)
    clave_delta = tuple(clave_delta_actual), tuple(clave_delta_efecto)

    idxs_alcance_delta = analizador.clave_submodular[EFFECT]
    dims_mecanismo_delta = analizador.clave_submodular[ACTUAL]

    if clave_delta not in analizador.memoria_delta:
        emd_delta, vector_delta_marginal = bipartir_y_emd(
            analizador, idxs_alcance_delta, dims_mecanismo_delta
        )
        analizador.memoria_delta[clave_delta] = emd_delta, vector_delta_marginal
    else:
        emd_delta, vector_delta_marginal = analizador.memoria_delta[clave_delta]

    for omega in omegas:
        analizador.definir_clave(omega)

    idxs_alcance_union = analizador.clave_submodular[EFFECT]
    dims_mecanismo_union = analizador.clave_submodular[ACTUAL]
    emd_union, _ = bipartir_y_emd(
        analizador, idxs_alcance_union, dims_mecanismo_union
    )
        
    return emd_union, emd_delta, vector_delta_marginal
