# evaluator.py

import numpy as np
from typing import List, Tuple
from src.constants.base import ACTUAL, EFFECT, INFTY_POS
from src.funcs.iit import emd_efecto, seleccionar_estado

def bloques_son_validos(blocks: list, k: int) -> bool:
    """Verifica si la partición tiene exactamente k bloques y ninguno tiene alcance futuro vacío."""
    if len(blocks) != k:
        return False
    for b in blocks:
        if not any(t == EFFECT for t, _ in b):
            return False
    return True

def get_biparticion_prob_variable(analizador, idx: int, alcance: List[int], mecanismo: List[int]) -> float:
    """
    Calcula la probabilidad marginal para una variable en una bipartición específica (alcance, mecanismo)
    aplicando la optimización "Slice-First" para evitar desbordamiento de memoria (OOM).

    El proceso se realiza en las siguientes fases:
    1. Identificación del NCubo:
       - Recupera el NCubo correspondiente al índice de la variable a partir del analizador del subsistema.
    2. Filtrado y Selección de Ejes (Slicing):
       - Si la variable está en el alcance, se seleccionan los ejes del mecanismo para rebanar (slicear).
       - Si no está en el alcance, se seleccionan los ejes fuera del mecanismo para rebanar.
    3. Reducción y Slicing Rápido:
       - Construye una tupla de rodajas (slices) donde las dimensiones activas del sistema se fijan al
         estado inicial y el resto se conserva completo.
       - Extrae la porción del array de datos (sliced_data) a velocidad sub-milisegundo sin duplicar memoria.
    4. Promedio Marginalizado:
       - Si quedan ejes sin rebanar en la vista resultante, calcula su media (np.mean) y retorna el valor.

    Args:
        analizador: Instancia del analizador que contiene el subsistema y el estado inicial.
        idx (int): Índice de la variable/dimensión a marginalizar.
        alcance (List[int]): Lista de índices de variables que conforman el alcance del bloque actual.
        mecanismo (List[int]): Lista de índices de variables que conforman el mecanismo del bloque actual.

    Returns:
        float: Probabilidad marginal calculada para la variable en la partición dada.
    """
    cubo = next((c for c in analizador.sia_subsistema.ncubos if c.indice == idx), None)
    if cubo is None:
        return 0.0

    mecanismo_set = set(mecanismo)

    if idx in alcance:
        # Se mantiene la conexión: slicear variables del mecanismo, promediar el resto
        ejes_a_slicear = [d for d in cubo.dims if d in mecanismo_set]
    else:
        # Se corta la conexión: promediar variables del mecanismo, slicear el resto
        ejes_a_slicear = [d for d in cubo.dims if d not in mecanismo_set]

    num_dims = cubo.dims.size
    seleccion = [slice(None)] * num_dims
    
    for dim in ejes_a_slicear:
        dim_idx = list(cubo.dims).index(dim)
        level_arr = num_dims - (dim_idx + 1)
        seleccion[level_arr] = analizador.sia_subsistema.estado_inicial[dim]

    # Slicing inicial ultra rápido (retorna una vista strided sin copiar datos)
    sliced_data = cubo.data[tuple(seleccion)]

    if sliced_data.ndim == 0:
        return float(sliced_data)
        
    # Promediar las dimensiones restantes
    return float(np.mean(sliced_data))

def bipartir_y_emd(
    analizador,
    idxs_alcance: list,
    dims_mecanismo: list,
    solo_emd: bool = False,
) -> tuple[float, np.ndarray]:
    """
    Realiza la bipartición de un bloque y calcula la distancia EMD correspondiente respecto a
    las distribuciones marginales del subsistema, utilizando una memoria caché interna.

    El proceso se realiza en las siguientes fases:
    1. Consulta de Caché:
       - Si solo se requiere la pérdida (solo_emd=True), busca en la caché bajo la clave de pérdida únicamente.
       - Si se requiere la distribución marginal completa, busca en la caché usando la tupla ordenada
         de alcances y mecanismos.
    2. Evaluación de Probabilidades:
       - Si la clave no está en caché, inicializa un vector de probabilidades marginales para todas las
         variables del subsistema.
       - Para cada variable, calcula la probabilidad marginal mediante la función optimizada Slice-First.
    3. Cálculo de la Distancia:
       - Ejecuta emd_efecto para obtener la diferencia entre las probabilidades calculadas y las
         marginales de referencia.
    4. Registro en Memoria:
       - Almacena el resultado (pérdida o tupla de pérdida y distribución) en la caché para evitar re-cálculos.

    Args:
        analizador: Instancia de la estrategia que provee el subsistema y almacena la caché.
        idxs_alcance (list): Lista de índices de variables pertenecientes al alcance del bloque.
        dims_mecanismo (list): Lista de dimensiones del mecanismo para la partición del bloque.
        solo_emd (bool, opcional): Si es True, calcula y retorna únicamente la pérdida EMD omitiendo
                                   la construcción del vector de distribución marginal completo en el retorno.

    Returns:
        tuple[float, np.ndarray]: Tupla que contiene:
            - La distancia EMD calculada.
            - El array con la distribución de probabilidad marginal (o None si solo_emd es True).
    """
    if not hasattr(analizador, "_bipartir_emd_cache"):
        analizador._bipartir_emd_cache = {}

    clave = (tuple(sorted(idxs_alcance)), tuple(sorted(dims_mecanismo)))

    if solo_emd:
        clave_emd = ("emd_only", *clave)
        if clave_emd not in analizador._bipartir_emd_cache:
            indices_sistema = list(analizador.sia_subsistema.indices_ncubos)
            hat_p = np.zeros(len(indices_sistema), dtype=np.float32)
            for pos, idx in enumerate(indices_sistema):
                hat_p[pos] = get_biparticion_prob_variable(analizador, idx, idxs_alcance, dims_mecanismo)
            emd = emd_efecto(hat_p, analizador.sia_dists_marginales)
            analizador._bipartir_emd_cache[clave_emd] = emd
        return analizador._bipartir_emd_cache[clave_emd], None

    if clave not in analizador._bipartir_emd_cache:
        indices_sistema = list(analizador.sia_subsistema.indices_ncubos)
        hat_p = np.zeros(len(indices_sistema), dtype=np.float32)
        for pos, idx in enumerate(indices_sistema):
            hat_p[pos] = get_biparticion_prob_variable(analizador, idx, idxs_alcance, dims_mecanismo)
        emd = emd_efecto(hat_p, analizador.sia_dists_marginales)
        analizador._bipartir_emd_cache[clave] = (emd, hat_p)

    return analizador._bipartir_emd_cache[clave]

def evaluate_k_partition(
    analizador, blocks: List[List[Tuple[int, int]]]
) -> Tuple[float, np.ndarray]:
    """
    Evalúa la pérdida total (EMD) y distribución resultante de una k-partición completa
    compuesta por múltiples bloques de variables independientes.

    El proceso se realiza en las siguientes fases:
    1. Validación y Filtro Inicial:
       - Elimina bloques vacíos y verifica que la partición sea válida (mínimo 2 bloques).
    2. Construcción de la Distribución Particionada:
       - Inicializa el vector de probabilidades hat_p para todo el subsistema.
       - Para cada bloque, extrae sus componentes de alcance y mecanismo.
       - Evalúa individualmente el bloque mediante bipartir_y_emd.
       - Asigna los valores de la distribución marginal del bloque a las posiciones del vector hat_p
         de las variables correspondientes a su alcance.
    3. Cálculo de Pérdida Global:
       - Calcula la EMD global entre el vector hat_p completo y las distribuciones de referencia del analizador.

    Args:
        analizador: Instancia de la estrategia que realiza la evaluación y contiene los metadatos del subsistema.
        blocks (List[List[Tuple[int, int]]]): Lista de bloques, donde cada bloque contiene tuplas
                                              identificadas por (tiempo, índice) de la variable.

    Returns:
        Tuple[float, np.ndarray]: Tupla que contiene:
            - Pérdida global EMD de la partición (o INFTY_POS si la partición es inválida).
            - Array con la distribución de probabilidad conjunta bajo la partición (hat_p).
    """
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
    """
    Evalúa el impacto de combinar el conjunto de nodos individual delta y su agrupación con el conjunto omega, calculando la diferencia entre EMD (Earth Mover's Distance) de las configuraciones, en conclusión los nodos delta evaluados individualmente y su combinación con el conjunto omega.

    El proceso se realiza en dos fases principales:

    1. Evaluación Individual:
       - Crea una copia del estado temporal del subsistema.
       - Activa los nodos delta en su tiempo correspondiente (presente/futuro).
       - Si el delta ya fue evaluado antes, recupera su EMD y distribución marginal de memoria
       - Si no, ha de:
         * Identificar dimensiones activas en presente y futuro.
         * Realiza bipartición del subsistema con esas dimensiones.
         * Calcular la distribución marginal y EMD respecto al subsistema.
         * Guarda resultados en memoria para seguro un uso futuro.

    2. Evaluación Combinada:
       - Sobre la misma copia temporal, activa también los nodos omega.
       - Calcula dimensiones activas totales (delta + omega).
       - Realiza bipartición del subsistema completo.
       - Obtiene EMD de la combinación.

    Args:
        deltas: Un nodo individual (tupla) o grupo de nodos (lista de tuplas)
               donde cada tupla está identificada por su (tiempo, índice), sea el tiempo t_0 identificado como 0, t_1 como 1 y, el índice hace referencia a las variables/dimensiones habilitadas para operaciones de substracción/marginalización sobre el subsistema, tal que genere la partición.
        omegas: Lista de nodos ya agrupados, puede contener tuplas individuales
               o listas de tuplas para grupos formados por los pares candidatos o más uniones entre sí (grupos candidatos).

    Returns:
        tuple: (
            EMD de la combinación omega y delta,
            EMD del delta individual,
            Distribución marginal del delta individual
        )
        Esto lo hice así para hacer almacenamiento externo de la emd individual y su distribución marginal en las particiones candidatas.
    """
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
