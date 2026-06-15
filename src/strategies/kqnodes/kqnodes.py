# kqnodes.py

import time
from typing import List, Tuple
import numpy as np
from src.strategies.kqnodes.tree_node import TreeNode
from src.strategies.q_nodes import QNodes
from src.models.core.solution import Solution
from src.constants.models import QNODES_LABEL, QNODES_ANALYSIS_TAG
from src.constants.base import ACTUAL, EFFECT, INT_ZERO, INFTY_POS, LAST_IDX, TYPE_TAG
from src.funcs.iit import LOWER_ABECEDARY, ABECEDARY
from src.funcs.format import fmt_particion_multi_k
from src.middlewares.profile import gestor_perfilado, profile

from src.strategies.kqnodes.evaluator import (
    evaluate_k_partition,
    funcion_submodular_k,
    bipartir_y_emd
)
from src.strategies.kqnodes.candidates import (
    generar_pool_candidatos,
    ejecutar_busqueda_exhaustiva,
    generar_particion_balanceada
)
from src.strategies.kqnodes.refinement import refinar_busqueda_local
from src.strategies.kqnodes.combinatorics import stirling_segundo_tipo
from src.strategies.kqnodes.tree_search import (
    build_tree_from_fusions,
    collect_internal_nodes,
    is_ancestor
)


class KQNodes(QNodes):
    """
    Extensión de QNodes para k-particiones (3 ≤ k ≤ 5) usando el árbol de fusiones.
    Conserva la estructura delegando a submódulos especializados para mantener la legibilidad.
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self.tree: TreeNode | None = None
        self.fusion_history: List = []

    def reset_estado(self):
        """Limpia todo el estado interno entre ejecuciones."""
        self.memoria_delta = {}
        self.memoria_grupo_candidato = {}
        self.memoria_union = {}
        self._bipartir_emd_cache = {}
        self.fusion_history = []
        self.tree = None

    # ====================== ÁRBOL ======================
    def _build_tree_from_fusions(self):
        return build_tree_from_fusions(self.fusion_history)

    def _collect_internal_nodes(self, node, res):
        return collect_internal_nodes(node, res)

    def _is_ancestor(self, ancestor, node):
        return is_ancestor(ancestor, node)

    # ====================== EVALUACIÓN (delegada) ======================
    def _evaluate_k_partition(self, blocks):
        return evaluate_k_partition(self, blocks)

    def _bipartir_y_emd(self, idxs_alcance, dims_mecanismo, solo_emd=False):
        return bipartir_y_emd(self, idxs_alcance, dims_mecanismo, solo_emd)

    def funcion_submodular_k(self, deltas, omegas):
        return funcion_submodular_k(self, deltas, omegas)

    # ====================== BÚSQUEDA Y CANDIDATOS (delegada) ======================
    def _generar_pool_candidatos(self, presente, futuro, k):
        return generar_pool_candidatos(self, presente, futuro, k)

    def _ejecutar_busqueda_exhaustiva(self, k):
        return ejecutar_busqueda_exhaustiva(self, k)

    def _generar_particion_balanceada(self, vertices, k):
        return generar_particion_balanceada(vertices, k)

    def _find_best_partition(self, candidates):
        """Evalúa todos los candidatos y retorna el mejor."""
        best_loss = INFTY_POS
        best_blocks = None
        best_dist = None

        for candidate in candidates:
            loss, dist = self._evaluate_k_partition(candidate)
            if loss < best_loss:
                best_loss = loss
                best_blocks = candidate
                best_dist = dist

        return best_blocks, best_loss, best_dist

    # ====================== REFINAMIENTO (delegada) ======================
    def _refinar_busqueda_local(self, blocks, k):
        return refinar_busqueda_local(self, blocks, k)

    # ====================== FORMATO ======================
    @staticmethod
    def fmt_particion_multi_k(
        particion_alcance: list[list[int]],
        particion_mecanismo: list[list[int]],
    ) -> str:
        linea_superior = ""
        linea_inferior = ""
        for alcance, mecanismo in zip(particion_alcance, particion_mecanismo):
            alcance = sorted(alcance)
            mecanismo = sorted(mecanismo)
            purv_str = ",".join(ABECEDARY[i] for i in alcance) if alcance else "∅"
            mech_str = (
                ",".join(LOWER_ABECEDARY[i] for i in mecanismo) if mecanismo else "∅"
            )
            width = max(len(purv_str), len(mech_str)) + 2
            linea_superior += f"⎛{purv_str:^{width}}⎞"
            linea_inferior += f"⎝{mech_str:^{width}}⎠"
        return f"{linea_superior}\n{linea_inferior}\n"

    # ====================== MÉTODO PRINCIPAL ======================
    def aplicar_estrategia_k(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 3,
    ) -> Solution:
        """K-partición usando el árbol de fusiones del algoritmo Q."""
        inicio = time.perf_counter()
        self.reset_estado()
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        # Caso degenerado: distribución trivial
        if np.allclose(self.sia_dists_marginales, 0.0):
            presente = [(ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos]
            futuro = [(EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos]
            vertices = []
            for a, e in zip(presente, futuro):
                vertices.append(a)
                vertices.append(e)
            vertices += (
                presente[len(futuro) :]
                if len(presente) > len(futuro)
                else futuro[len(presente) :]
            )
            blocks = self._generar_particion_balanceada(vertices, k)
            particion_alcance = [[idx for t, idx in b if t == EFFECT] for b in blocks]
            particion_mecanismo = [[idx for t, idx in b if t == ACTUAL] for b in blocks]
            fmt = self.fmt_particion_multi_k(particion_alcance, particion_mecanismo)
            return Solution(
                estrategia=f"KQNodes(k={k})-TrivialZero",
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales,
                particion=fmt,
                tiempo_total=time.perf_counter() - inicio,
            )

        presente = [(ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos]
        futuro = [(EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos]

        # ── BÚSQUEDA EXHAUSTIVA O HEURÍSTICA SEGÚN TAMAÑO ─────────────────────
        n_nodos = len(self.sia_subsistema.indices_ncubos)
        n_pres = len(self.sia_subsistema.dims_ncubos)
        num_comb = stirling_segundo_tipo(n_nodos, k) * (k ** n_pres)
        
        if k <= n_nodos <= 6 and num_comb <= 500:
            print(f"   [Exhaustivo] Ejecutando búsqueda exhaustiva para N={n_nodos} (futuro) y N_pres={n_pres} (presente) | Combinaciones: {num_comb} | k={k}...")
            best_blocks, best_loss, best_dist = self._ejecutar_busqueda_exhaustiva(k)
        else:
            pool = self._generar_pool_candidatos(presente, futuro, k)
            best_blocks, best_loss, best_dist = self._find_best_partition(pool)

            if best_blocks is None:
                best_blocks = pool[0]
                best_loss = INFTY_POS
                best_dist = self.sia_dists_marginales
            else:
                # Aplicar refinamiento por búsqueda local (Hill Climbing / VND)
                print(f"   [Refinamiento] Pérdida antes: {best_loss:.6f}")
                best_blocks, best_loss, best_dist = self._refinar_busqueda_local(best_blocks, k)
                print(f"   [Refinamiento] Pérdida después de Búsqueda Local: {best_loss:.6f}")

        particion_alcance = [[idx for t, idx in b if t == EFFECT] for b in best_blocks]
        particion_mecanismo = [
            [idx for t, idx in b if t == ACTUAL] for b in best_blocks
        ]
        fmt = self.fmt_particion_multi_k(particion_alcance, particion_mecanismo)

        return Solution(
            estrategia=f"KQNodes(k={k})",
            perdida=best_loss,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=(
                best_dist if best_dist is not None else self.sia_dists_marginales
            ),
            particion=fmt,
            tiempo_total=time.perf_counter() - inicio,
        )

    # ====================== ALGORITMO Q BASE ======================
    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def algorithm(self, vertices: list[tuple[int, int]]):
        indice_emd = INT_ZERO

        for i in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]
            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for j in range(len(deltas_ciclo) - 1):
                emd_local = 1e5
                indice_mip: int

                for idx_k in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular_k(
                        deltas_ciclo[idx_k], omegas_ciclo
                    )
                    emd_iteracion = emd_union - emd_delta

                    if emd_iteracion < emd_local:
                        if emd_delta == INT_ZERO:
                            clave = (
                                tuple(deltas_ciclo[idx_k])
                                if isinstance(deltas_ciclo[idx_k], list)
                                else (deltas_ciclo[idx_k],)
                            )
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_marginal_delta,
                            )
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = idx_k
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            self.memoria_grupo_candidato[
                tuple(
                    deltas_ciclo[LAST_IDX]
                    if isinstance(deltas_ciclo[LAST_IDX], list)
                    else deltas_ciclo
                )
            ] = (emd_particion_candidata, dist_particion_candidata)

            par_candidato = (
                [omegas_ciclo[LAST_IDX]]
                if isinstance(omegas_ciclo[LAST_IDX], tuple)
                else omegas_ciclo[LAST_IDX]
            ) + (
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )

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
            omegas_ciclo.append(par_candidato)
            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k][indice_emd],
        )
