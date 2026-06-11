"""
kqnodes_optimizado.py
─────────────────────
Reemplaza tu archivo kqnodes.py actual con este.

OPTIMIZACIONES APLICADAS:
  1. definir_clave_pura      → sin side-effects, sin tocar self.clave_submodular
  2. funcion_submodular      → thread-safe, usa definir_clave_pura internamente
  3. algorithm               → early exit cuando emd_local == 0 o emd_delta == 0
  4. algorithm               → loop k paralelizado con ThreadPoolExecutor
  5. KQNodes.__init__        → inicializa _lock y _executor una sola vez

CÓMO MIGRAR:
  - Copia esta clase y reemplaza KQNodes en tu archivo actual.
  - QNodes (padre) no cambia.
  - El resto del proyecto no necesita cambios.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union
import numpy as np
from itertools import combinations

from src.models.kqnodes.tree_node import TreeNode
from src.strategies.q_nodes import QNodes
from src.models.core.solution import Solution
from src.constants.models import QNODES_LABEL
from src.constants.base import ACTUAL, EFFECT, INT_ZERO, INFTY_POS, LAST_IDX, TYPE_TAG
from src.funcs.iit import LOWER_ABECEDARY, ABECEDARY, emd_efecto
from src.funcs.format import fmt_particion_multi_k
from src.middlewares.profile import gestor_perfilado, profile
from src.constants.models import QNODES_ANALYSIS_TAG


class KQNodes(QNodes):
    """
    Extensión de QNodes para k-particiones (3 ≤ k ≤ 5).
    Versión optimizada con paralelismo y early exit.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self.tree: TreeNode | None = None
        self.fusion_history: List = []

        # ── NUEVO: estructuras para thread-safety ──────────────────────────────
        # _lock protege memoria_delta cuando múltiples threads escriben en paralelo
        self._lock = threading.Lock()
        # _executor se crea una vez y se reutiliza en todas las llamadas a algorithm
        # max_workers=4 es conservador; puedes subir a 8 si tu CPU tiene más núcleos
        self._executor = ThreadPoolExecutor(max_workers=4)

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMIZACIÓN 1: definir_clave_pura
    # ──────────────────────────────────
    # PROBLEMA ORIGINAL:
    #   definir_clave() escribe en self.clave_submodular (lista compartida).
    #   Si dos threads llaman a funcion_submodular al mismo tiempo, ambos
    #   modifican self.clave_submodular y los resultados se corrompen.
    #
    # SOLUCIÓN:
    #   Esta versión NO toca self. Calcula y retorna los índices localmente.
    #   Es equivalente a la original pero sin side-effects.
    # ══════════════════════════════════════════════════════════════════════════
    def _definir_clave_pura(
        self,
        conjunto: Union[tuple, list],
    ) -> tuple[list[int], list[int]]:
        """
        Versión pura (sin side-effects) de definir_clave.

        Retorna:
            (lista_ACTUAL, lista_EFFECT)  ← índices ordenados, sin tocar self
        """
        actuales: list[int] = []
        efectos:  list[int] = []

        if isinstance(conjunto, tuple) and len(conjunto) == 2:
            # Caso: un solo vértice, e.g. (ACTUAL, 3)
            tiempo, indice = conjunto
            if tiempo == ACTUAL:
                actuales.append(indice)
            else:
                efectos.append(indice)

        elif isinstance(conjunto, list):
            for item in conjunto:
                if isinstance(item, tuple) and len(item) == 2:
                    tiempo, indice = item
                    if tiempo == ACTUAL:
                        actuales.append(indice)
                    else:
                        efectos.append(indice)
                elif isinstance(item, list):
                    # Grupo anidado (par candidato fusionado)
                    sub_act, sub_eff = self._definir_clave_pura(item)
                    actuales += sub_act
                    efectos  += sub_eff

        actuales.sort()
        efectos.sort()
        return actuales, efectos

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMIZACIÓN 2: funcion_submodular thread-safe
    # ───────────────────────────────────────────────
    # PROBLEMA ORIGINAL:
    #   Usaba self.clave_submodular como buffer compartido → no paralelizable.
    #   El lock en memoria_delta faltaba → race condition al escribir el cache.
    #
    # SOLUCIÓN:
    #   - Usa _definir_clave_pura (local, sin self)
    #   - Lock solo en la escritura de memoria_delta (lectura no necesita lock
    #     porque dict.__contains__ y dict.__getitem__ son atómicos en CPython)
    #   - Lógica idéntica a la original, solo cambia dónde viven las listas
    # ══════════════════════════════════════════════════════════════════════════
    def funcion_submodular(
        self,
        deltas: Union[tuple, list[tuple]],
        omegas: list[Union[tuple, list[tuple]]],
    ) -> tuple[float, float, np.ndarray]:
        """
        Thread-safe. Calcula:
            (emd_union, emd_delta, distribucion_marginal_delta)
        """

        # ── Parte 1: Delta individual ─────────────────────────────────────────
        dims_mecanismo_delta, idxs_alcance_delta = self._definir_clave_pura(deltas)
        #                     ↑ ACTUAL                ↑ EFFECT
        clave_delta = (tuple(dims_mecanismo_delta), tuple(idxs_alcance_delta))

        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(idxs_alcance_delta,   dtype=np.int8),
                np.array(dims_mecanismo_delta, dtype=np.int8),
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)

            # ── Lock solo al escribir en el cache compartido ──────────────────
            with self._lock:
                # Doble chequeo: otro thread pudo haber escrito mientras esperábamos
                if clave_delta not in self.memoria_delta:
                    self.memoria_delta[clave_delta] = (emd_delta, vector_delta_marginal)
        else:
            emd_delta, vector_delta_marginal = self.memoria_delta[clave_delta]

        # ── Parte 2: Unión (delta + todos los omegas) ─────────────────────────
        # Acumulamos localmente sin tocar self
        alcance_union:   list[int] = list(idxs_alcance_delta)
        mecanismo_union: list[int] = list(dims_mecanismo_delta)

        for omega in omegas:
            act, eff = self._definir_clave_pura(omega)
            mecanismo_union += act
            alcance_union   += eff

        # Deduplicar y ordenar (puede haber índices repetidos en grupos fusionados)
        alcance_union   = sorted(set(alcance_union))
        mecanismo_union = sorted(set(mecanismo_union))

        particion_union = self.sia_subsistema.bipartir(
            np.array(alcance_union,   dtype=np.int8),
            np.array(mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)

        return emd_union, emd_delta, vector_delta_marginal

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMIZACIÓN 3 + 4: algorithm con early exit y loop k paralelizado
    # ────────────────────────────────────────────────────────────────────
    # PROBLEMA ORIGINAL:
    #   El loop k (evaluación de cada delta candidato) era secuencial.
    #   No había salida anticipada cuando ya no podía mejorar.
    #
    # SOLUCIÓN:
    #   3. Early exit: si emd_local llega a 0, no hay partición mejor → break
    #   4. Loop k paralelo: cada delta se evalúa en un thread separado.
    #      Como NumPy libera el GIL, ThreadPoolExecutor da speedup real.
    #
    # IMPORTANTE: omegas_ciclo se pasa como copia (omegas_ciclo[:]) para
    #   que cada thread vea el mismo estado sin que otro lo modifique.
    # ══════════════════════════════════════════════════════════════════════════
    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def algorithm(self, vertices: list[tuple[int, int]]):
        """
        Algoritmo Q optimizado.
        Cambios respecto al original:
          - Loop k evaluado en paralelo (4 workers)
          - Early exit si emd_local == 0 dentro del loop j
          - Early exit si emd_particion_candidata == 0 al final de fase i
        El resto de la lógica es idéntica al original.
        """
        indice_emd = INT_ZERO

        for i in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for j in range(len(deltas_ciclo) - 1):

                emd_local = 1e5
                indice_mip: int = 0

                # ── OPTIMIZACIÓN 4: evaluar todos los deltas en paralelo ───────
                # Capturamos snapshot de omegas para que todos los threads
                # vean el mismo estado (inmutable durante este loop k)
                omegas_snapshot = omegas_ciclo[:]

                def evaluar(k_idx: int, omegas=omegas_snapshot):
                    """Función que corre en cada thread."""
                    return k_idx, self.funcion_submodular(
                        deltas_ciclo[k_idx], omegas
                    )

                futures = {
                    self._executor.submit(evaluar, k): k
                    for k in range(len(deltas_ciclo))
                }

                for future in as_completed(futures):
                    k_idx, (emd_union, emd_delta, dist_marginal_delta) = future.result()
                    emd_iteracion = emd_union - emd_delta

                    if emd_iteracion < emd_local:

                        # Early exit original: delta con emd=0 es partición perfecta
                        if emd_delta == INT_ZERO:
                            clave = (
                                tuple(deltas_ciclo[k_idx])
                                if isinstance(deltas_ciclo[k_idx], list)
                                else (deltas_ciclo[k_idx],)
                            )
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_marginal_delta,
                            )
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = k_idx
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                # ── OPTIMIZACIÓN 3: early exit en loop j ──────────────────────
                # Si la mejor diferencia encontrada es 0, ningún otro delta
                # en este ciclo j puede mejorar → salimos del loop j
                if emd_local == 0.0:
                    break

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            # Guardamos el candidato de esta fase
            self.memoria_grupo_candidato[
                tuple(
                    deltas_ciclo[LAST_IDX]
                    if isinstance(deltas_ciclo[LAST_IDX], list)
                    else deltas_ciclo
                )
            ] = (emd_particion_candidata, dist_particion_candidata)

            # ── OPTIMIZACIÓN 3: early exit en loop i (fases) ──────────────────
            # Si ya encontramos una partición con pérdida 0, no tiene sentido
            # seguir explorando más fases
            if emd_particion_candidata == 0.0:
                break

            # Fusión (igual al original)
            left_group = (
                [omegas_ciclo[-1]]
                if isinstance(omegas_ciclo[-1], tuple)
                else omegas_ciclo[-1].copy()
            )
            right_group = (
                [deltas_ciclo[-1]]
                if isinstance(deltas_ciclo[-1], tuple)
                else deltas_ciclo[-1].copy()
            )
            fusion_group = left_group + right_group

            self.fusion_history.append((left_group, right_group, fusion_group))

            omegas_ciclo.pop()
            omegas_ciclo.append(left_group + right_group)
            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k][indice_emd],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Resto de métodos: sin cambios respecto a tu versión que ya funciona
    # ══════════════════════════════════════════════════════════════════════════

    def _build_tree_from_fusions(self):
        node_map = {}
        root = None
        for left, right, fusion in self.fusion_history:
            left_key   = tuple(sorted(left))
            right_key  = tuple(sorted(right))
            fusion_key = tuple(sorted(fusion))

            if left_key not in node_map:
                node_map[left_key] = TreeNode(sorted(left))
            if right_key not in node_map:
                node_map[right_key] = TreeNode(sorted(right))
            if fusion_key not in node_map:
                parent = TreeNode(sorted(fusion))
                parent.left  = node_map[left_key]
                parent.right = node_map[right_key]
                node_map[left_key].parent  = parent
                node_map[right_key].parent = parent
                node_map[fusion_key] = parent
            root = node_map[fusion_key]
        return root

    def _collect_internal_nodes(self, node, res):
        if node is None:
            return
        if node.left and node.right:
            res.append(node)
        self._collect_internal_nodes(node.left, res)
        self._collect_internal_nodes(node.right, res)

    def _generate_k_cuts(self, k: int):
        internal = []
        self._collect_internal_nodes(self.tree, internal)
        candidates = []
        seen = set()

        for cut_combo in combinations(internal, k - 1):
            cut_set = set(cut_combo)
            blocks = []

            def dfs(node):
                if node is None:
                    return
                if node in cut_set or node.is_leaf:
                    leaves = node.get_all_leaves()
                    if leaves:
                        blocks.append(sorted(leaves))
                    return
                dfs(node.left)
                dfs(node.right)

            dfs(self.tree)
            valid_blocks = [b for b in blocks if len(b) > 0]

            if len(valid_blocks) == k:
                signature = tuple(sorted(tuple(b) for b in valid_blocks))
                if signature not in seen:
                    seen.add(signature)
                    candidates.append(valid_blocks)

        if len(candidates) < 5:
            candidates.append(self._generar_particion_balanceada(
                sorted(self.tree.get_all_leaves(), key=lambda x: (x[0], x[1])), k
            ))

        return candidates[:25]

    def _bloques_son_validos(self, blocks: list, k: int) -> bool:
        if len(blocks) != k:
            return False
        for b in blocks:
            if not any(t == ACTUAL for t, _ in b):
                return False
            if not any(t == EFFECT for t, _ in b):
                return False
        return True

    def _generar_particion_balanceada(self, vertices: list, k: int) -> list[list]:
        bloques = [[] for _ in range(k)]
        for i, v in enumerate(vertices):
            bloques[i % k].append(v)

        for i, bloque in enumerate(bloques):
            if not any(t == ACTUAL for t, _ in bloque):
                for j, otro in enumerate(bloques):
                    actuales = [(t, idx) for t, idx in otro if t == ACTUAL]
                    if len(actuales) >= 2:
                        v_robar = actuales[-1]
                        otro.remove(v_robar)
                        bloque.append(v_robar)
                        break
            if not any(t == EFFECT for t, _ in bloque):
                for j, otro in enumerate(bloques):
                    effects = [(t, idx) for t, idx in otro if t == EFFECT]
                    if len(effects) >= 2:
                        v_robar = effects[-1]
                        otro.remove(v_robar)
                        bloque.append(v_robar)
                        break
        return bloques

    def _evaluate_k_partition(
        self, blocks: List[List[Tuple[int, int]]]
    ) -> Tuple[float, np.ndarray]:
        blocks = [b for b in blocks if len(b) > 0]
        if len(blocks) < 2:
            return INFTY_POS, None

        particion_alcance   = [[idx for t, idx in b if t == EFFECT] for b in blocks]
        particion_mecanismo = [[idx for t, idx in b if t == ACTUAL]  for b in blocks]

        try:
            sistema_part = self.sia_subsistema.particionar_multi_k(
                particion_alcance, particion_mecanismo
            )
            hat_p = sistema_part.distribucion_marginal()
            p_v   = self.sia_dists_marginales

            if len(hat_p) != len(p_v):
                return INFTY_POS, None

            loss = emd_efecto(hat_p, p_v)
            return loss, hat_p
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return INFTY_POS, None

    def _find_best_partition(self, candidates):
        best_loss   = INFTY_POS
        best_blocks = None
        best_dist   = None
        for candidate in candidates:
            loss, dist = self._evaluate_k_partition(candidate)
            if loss < best_loss:
                best_loss   = loss
                best_blocks = candidate
                best_dist   = dist
        return best_blocks, best_loss, best_dist

    @staticmethod
    def fmt_particion_multi_k(
        particion_alcance:   list[list[int]],
        particion_mecanismo: list[list[int]],
    ) -> str:
        linea_superior = ""
        linea_inferior = ""
        for alcance, mecanismo in zip(particion_alcance, particion_mecanismo):
            alcance   = sorted(alcance)
            mecanismo = sorted(mecanismo)
            purv_str = ",".join(ABECEDARY[i]       for i in alcance)   if alcance   else "∅"
            mech_str = ",".join(LOWER_ABECEDARY[i] for i in mecanismo) if mecanismo else "∅"
            width = max(len(purv_str), len(mech_str)) + 2
            linea_superior += f"⎛{purv_str:^{width}}⎞"
            linea_inferior += f"⎝{mech_str:^{width}}⎠"
        return f"{linea_superior}\n{linea_inferior}\n"

    # ══════════════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL: aplicar_estrategia_k
    # Sin cambios de lógica — solo usa los métodos optimizados de arriba
    # ══════════════════════════════════════════════════════════════════════════
    def aplicar_estrategia_k(
        self,
        estado_inicial: str,
        condicion:      str,
        alcance:        str,
        mecanismo:      str,
        k:              int = 3,
    ) -> Solution:
        inicio = time.perf_counter()
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        presente = [(ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos]
        futuro   = [(EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos]

        # Intercalar ACTUAL y EFFECT para que los cortes del árbol
        # repartan ambos tipos en cada bloque (evita ∅)
        vertices = []
        for a, e in zip(presente, futuro):
            vertices.append(a)
            vertices.append(e)
        if len(presente) > len(futuro):
            vertices += presente[len(futuro):]
        elif len(futuro) > len(presente):
            vertices += futuro[len(presente):]

        # Construir árbol con el algoritmo Q
        self.fusion_history.clear()
        self.algorithm(vertices)
        self.tree = self._build_tree_from_fusions()

        if self.tree is None:
            return self._fallback_solution(k, inicio)

        # Generar y filtrar candidatos
        candidates = self._generate_k_cuts(k)
        candidates_validos = [c for c in candidates if self._bloques_son_validos(c, k)]

        if not candidates_validos:
            candidates_validos = [self._generar_particion_balanceada(vertices, k)]

        # Evaluar
        best_blocks, best_loss, best_dist = self._find_best_partition(candidates_validos)

        if best_blocks is None:
            best_blocks = candidates_validos[0]
            best_loss   = INFTY_POS
            best_dist   = self.sia_dists_marginales

        particion_alcance   = [[idx for t, idx in b if t == EFFECT] for b in best_blocks]
        particion_mecanismo = [[idx for t, idx in b if t == ACTUAL]  for b in best_blocks]
        fmt = self.fmt_particion_multi_k(particion_alcance, particion_mecanismo)

        return Solution(
            estrategia=f"KQNodes(k={k})",
            perdida=best_loss,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=best_dist if best_dist is not None else self.sia_dists_marginales,
            particion=fmt,
            tiempo_total=time.perf_counter() - inicio,
        )

    def _fallback_solution(self, k: int, inicio: float) -> Solution:
        presente = [(ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos]
        futuro   = [(EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos]
        vertices = []
        for a, e in zip(presente, futuro):
            vertices.append(a)
            vertices.append(e)
        vertices += presente[len(futuro):] if len(presente) > len(futuro) else futuro[len(presente):]

        blocks = self._generar_particion_balanceada(vertices, k)
        particion_alcance   = [[idx for t, idx in b if t == EFFECT] for b in blocks]
        particion_mecanismo = [[idx for t, idx in b if t == ACTUAL]  for b in blocks]
        fmt = self.fmt_particion_multi_k(particion_alcance, particion_mecanismo)

        return Solution(
            estrategia=f"KQNodes-Fallback(k={k})",
            perdida=INFTY_POS,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.sia_dists_marginales,
            particion=fmt,
            tiempo_total=time.perf_counter() - inicio,
        )

    # definir_clave original se mantiene para compatibilidad con QNodes padre
    def definir_clave(
        self,
        conjunto: Union[tuple, list],
    ):
        """
        Original — solo se usa en aplicar_estrategia (k=2, clase padre QNodes).
        Para k>2 usamos _definir_clave_pura internamente.
        """
        if isinstance(conjunto, tuple):
            tiempo, indice = conjunto
            self.clave_submodular[tiempo].append(indice)
        else:
            for tiempo, indice in conjunto:
                self.clave_submodular[tiempo].append(indice)
        self.clave_submodular[ACTUAL].sort()
        self.clave_submodular[EFFECT].sort()
        return self.clave_submodular