import time
from typing import List, Tuple
import numpy as np
from itertools import combinations
from src.models.kqnodes.tree_node import TreeNode
from src.strategies.q_nodes import QNodes
from src.models.core.solution import Solution
from src.constants.models import QNODES_LABEL, QNODES_ANALYSIS_TAG
from src.constants.base import ACTUAL, EFFECT, INT_ZERO, INFTY_POS, LAST_IDX, TYPE_TAG
from src.funcs.iit import LOWER_ABECEDARY, emd_efecto, ABECEDARY
from src.funcs.format import fmt_particion_multi_k
from src.middlewares.profile import gestor_perfilado, profile


class KQNodes(QNodes):
    """
    Extensión de QNodes para k-particiones (3 ≤ k ≤ 5) usando el árbol de fusiones.
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
        node_map = {}
        root = None
        for left, right, fusion in self.fusion_history:
            left_key  = tuple(sorted(left))
            right_key = tuple(sorted(right))
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
        self._collect_internal_nodes(node.left,  res)
        self._collect_internal_nodes(node.right, res)

    def _is_ancestor(self, ancestor, node):
        current = node.parent
        while current is not None:
            if current is ancestor:
                return True
            current = current.parent
        return False

    # ====================== EVALUACIÓN ======================

    def _evaluate_k_partition(
        self, blocks: List[List[Tuple[int, int]]]
    ) -> Tuple[float, np.ndarray]:
        """Evalúa una k-partición usando particionar_multi_k."""
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
            print(f"[_evaluate_k_partition] Error: {e}")
            return INFTY_POS, None

    def _find_best_partition(self, candidates):
        """Evalúa todos los candidatos y retorna el mejor."""
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

    # ====================== GENERACIÓN DE CANDIDATOS ======================

    def _q_particiones(self, indices: list, k: int) -> list[list[list[int]]]:
        """
        Corre el algoritmo Q sobre una lista de índices y extrae
        múltiples particiones en k grupos desde el árbol resultante.
        """
        if len(indices) < k:
            return []

        vertices = [(ACTUAL, idx) for idx in indices]

        # Backup del estado para no contaminarlo
        hist_backup = self.fusion_history.copy()
        mem_backup  = dict(self.memoria_grupo_candidato)
        self.fusion_history.clear()
        self.memoria_grupo_candidato.clear()

        try:
            self.algorithm(vertices)
            arbol = self._build_tree_from_fusions()
            if arbol is None:
                return []

            internal = []
            self._collect_internal_nodes(arbol, internal)
            if len(internal) < k - 1:
                return []

            particiones = []
            seen_part   = set()

            for cut_combo in combinations(internal, k - 1):
                cut_set = set(cut_combo)
                # ── Fix closure bug: pasar cut_set como argumento por defecto ──
                grupos = []
                def dfs(node, cut_set=cut_set, grupos=grupos):
                    if node is None:
                        return
                    if node in cut_set or node.is_leaf:
                        leaves = node.get_all_leaves()
                        idxs = [idx for _, idx in leaves]
                        if idxs:
                            grupos.append(sorted(idxs))
                        return
                    dfs(node.left,  cut_set, grupos)
                    dfs(node.right, cut_set, grupos)

                dfs(arbol)

                if len(grupos) == k and all(len(g) > 0 for g in grupos):
                    sig = tuple(sorted(tuple(g) for g in grupos))
                    if sig not in seen_part:
                        seen_part.add(sig)
                        particiones.append(grupos)

                if len(particiones) >= 10:
                    break

            return particiones

        except Exception as e:
            print(f"[_q_particiones] Error: {e}")
            return []
        finally:
            # Siempre restaurar el estado original
            self.fusion_history         = hist_backup
            self.memoria_grupo_candidato = mem_backup

    def _particiones_estructurales(self, indices: list, k: int) -> list[list[list[int]]]:
        """
        Genera particiones estructurales: bloques contiguos y round-robin
        con distintos offsets, más una partición por valor de distribución marginal.
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

        # Bloques contiguos con distintos offsets
        tam = max(1, n // k)
        for offset in range(k):
            rotado = indices[offset:] + indices[:offset]
            grupos = [sorted(rotado[i * tam:(i + 1) * tam]) for i in range(k)]
            sobrantes = rotado[k * tam:]
            for i, idx in enumerate(sobrantes):
                grupos[i % k].append(idx)
                grupos[i % k].sort()
            agregar(grupos)

        # Round-robin con distintos puntos de inicio
        for offset in range(min(k * 3, n)):
            rotado = indices[offset:] + indices[:offset]
            grupos = [[] for _ in range(k)]
            for i, idx in enumerate(rotado):
                grupos[i % k].append(idx)
            agregar(grupos)

        # Por valor de distribución marginal
        n_dist = len(self.sia_dists_marginales)
        if indices and max(indices, default=0) < n_dist:
            indices_por_dist = sorted(
                indices,
                key=lambda i: self.sia_dists_marginales[i]
            )
            grupos = [[] for _ in range(k)]
            for i, idx in enumerate(indices_por_dist):
                grupos[i % k].append(idx)
            agregar(grupos)

        return particiones

    def _generar_pool_candidatos(self, presente, futuro, k):
        """
        Genera un pool diverso de candidatos combinando particiones
        independientes de alcance (futuros) y mecanismo (presentes).
        """
        indices_futuros   = sorted(set(idx for _, idx in futuro))
        indices_presentes = sorted(set(idx for _, idx in presente))

        map_effect = {idx: (EFFECT, idx) for idx in indices_futuros}
        map_actual = {idx: (ACTUAL, idx) for idx in indices_presentes}

        pool = []
        seen = set()

        def registrar(part_alcance, part_mec):
            if len(part_alcance) != k or len(part_mec) != k:
                return False
            blocks = []
            for alc, mec in zip(part_alcance, part_mec):
                bloque  = [map_effect[idx] for idx in alc if idx in map_effect]
                bloque += [map_actual[idx] for idx in mec if idx in map_actual]
                if bloque:
                    blocks.append(bloque)
            if not self._bloques_son_validos(blocks, k):
                return False
            sig = tuple(sorted(tuple(sorted(b)) for b in blocks))
            if sig in seen:
                return False
            seen.add(sig)
            pool.append(blocks)
            return True

        # Q independiente sobre futuros y presentes
        parts_futuros   = self._q_particiones(indices_futuros,   k)
        parts_presentes = self._q_particiones(indices_presentes, k)

        for p_alc in parts_futuros:
            for p_mec in parts_presentes:
                registrar(p_alc, p_mec)

        print(f"→ Candidatos tras Q×Q: {len(pool)}")

        # Particiones estructurales combinadas con Q
        parts_struct_f = self._particiones_estructurales(indices_futuros,   k)
        parts_struct_p = self._particiones_estructurales(indices_presentes, k)

        for p_alc in parts_futuros + parts_struct_f:
            for p_mec in parts_presentes + parts_struct_p:
                registrar(p_alc, p_mec)

        print(f"→ Total candidatos en pool: {len(pool)}")

        # Garantizar al menos 1
        if not pool:
            grupos_f = [indices_futuros[i::k]   for i in range(k)]
            grupos_p = [indices_presentes[i::k]  for i in range(k)]
            registrar(grupos_f, grupos_p)

        return pool[:50]

    # ====================== HELPERS ======================

    def _bloques_son_validos(self, blocks: list, k: int) -> bool:
        """Verifica que haya exactamente k bloques, cada uno con ≥1 ACTUAL y ≥1 EFFECT."""
        if len(blocks) != k:
            return False
        for b in blocks:
            if not any(t == ACTUAL for t, _ in b):
                return False
            if not any(t == EFFECT for t, _ in b):
                return False
        return True

    def _generar_particion_balanceada(self, vertices: list, k: int) -> list[list]:
        """Round-robin garantizando ACTUAL y EFFECT en cada bloque."""
        bloques = [[] for _ in range(k)]
        for i, v in enumerate(vertices):
            bloques[i % k].append(v)

        for i, bloque in enumerate(bloques):
            if not any(t == ACTUAL for t, _ in bloque):
                for otro in bloques:
                    actuales = [(t, idx) for t, idx in otro if t == ACTUAL]
                    if len(actuales) >= 2:
                        v_robar = actuales[-1]
                        otro.remove(v_robar)
                        bloque.append(v_robar)
                        break
            if not any(t == EFFECT for t, _ in bloque):
                for otro in bloques:
                    effects = [(t, idx) for t, idx in otro if t == EFFECT]
                    if len(effects) >= 2:
                        v_robar = effects[-1]
                        otro.remove(v_robar)
                        bloque.append(v_robar)
                        break
        return bloques

    def _fallback_solution(self, k: int, inicio: float) -> Solution:
        """Solución de emergencia cuando el árbol no se puede construir."""
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

    # ====================== FORMATO ======================

    @staticmethod
    def fmt_particion_multi_k(
        particion_alcance:   list[list[int]],
        particion_mecanismo: list[list[int]],
    ) -> str:
        linea_superior = ""
        linea_inferior = ""
        for alcance, mecanismo in zip(particion_alcance, particion_mecanismo):
            alcance  = sorted(alcance)
            mecanismo = sorted(mecanismo)
            purv_str = ",".join(ABECEDARY[i]       for i in alcance)   if alcance   else "∅"
            mech_str = ",".join(LOWER_ABECEDARY[i] for i in mecanismo) if mecanismo else "∅"
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
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        # Caso degenerado: distribución trivial
        if np.allclose(self.sia_dists_marginales, 0.0):
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
                estrategia=f"KQNodes(k={k})-TrivialZero",
                perdida=0.0,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.sia_dists_marginales,
                particion=fmt,
                tiempo_total=time.perf_counter() - inicio,
            )

        presente = [(ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos]
        futuro   = [(EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos]

        pool = self._generar_pool_candidatos(presente, futuro, k)
        best_blocks, best_loss, best_dist = self._find_best_partition(pool)

        if best_blocks is None:
            best_blocks = pool[0]
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

    # ====================== ALGORITMO Q BASE ======================

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def algorithm(self, vertices: list[tuple[int, int]]):
        # ... (sin cambios, igual que el original)
        indice_emd = INT_ZERO

        for i in range(len(vertices) - 1):
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]
            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for j in range(len(deltas_ciclo) - 1):
                emd_local = 1e5
                indice_mip: int

                for k in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                        deltas_ciclo[k], omegas_ciclo
                    )
                    emd_iteracion = emd_union - emd_delta

                    if emd_iteracion < emd_local:
                        if emd_delta == INT_ZERO:
                            clave = (
                                tuple(deltas_ciclo[k])
                                if isinstance(deltas_ciclo[k], list)
                                else (deltas_ciclo[k],)
                            )
                            self.memoria_grupo_candidato[clave] = (emd_delta, dist_marginal_delta)
                            return clave

                        emd_local    = emd_iteracion
                        indice_mip   = k
                        emd_particion_candidata  = emd_delta
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

            left_group  = ([omegas_ciclo[-1]] if isinstance(omegas_ciclo[-1], tuple) else omegas_ciclo[-1].copy())
            right_group = ([deltas_ciclo[-1]]  if isinstance(deltas_ciclo[-1], tuple) else deltas_ciclo[-1].copy())
            fusion_group = left_group + right_group

            self.fusion_history.append((left_group, right_group, fusion_group))

            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)
            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k][indice_emd],
        )