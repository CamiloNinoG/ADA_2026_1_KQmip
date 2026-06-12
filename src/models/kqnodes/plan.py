import time
from typing import List, Tuple
import numpy as np
from itertools import combinations
from src.models.kqnodes.tree_node import TreeNode
from src.strategies.q_nodes import QNodes
from src.models.core.solution import Solution
from src.constants.models import QNODES_LABEL
from src.constants.base import ACTUAL, EFFECT, INT_ZERO, INFTY_POS
from src.funcs.iit import LOWER_ABECEDARY, emd_efecto
from src.funcs.format import fmt_particion_multi_k
import time
import numpy as np
from src.funcs.iit import emd_efecto, ABECEDARY
from src.middlewares.profile import gestor_perfilado, profile

from src.models.core.solution import Solution
from src.constants.models import QNODES_ANALYSIS_TAG
from src.constants.base import (
    INT_ZERO,
    TYPE_TAG,
    INFTY_POS,
    LAST_IDX,
    EFFECT,
    ACTUAL,
)


class KQNodes(QNodes):
    """
    Extensión de QNodes para k-particiones (3 ≤ k ≤ 5) usando el árbol de fusiones.
    """

    def _build_tree_from_fusions(self):
        node_map = {}
        root = None

        for left, right, fusion in self.fusion_history:

            left_key = tuple(sorted(left))
            right_key = tuple(sorted(right))
            fusion_key = tuple(sorted(fusion))

            if left_key not in node_map:
                node_map[left_key] = TreeNode(sorted(left))

            if right_key not in node_map:
                node_map[right_key] = TreeNode(sorted(right))

            if fusion_key not in node_map:

                parent = TreeNode(sorted(fusion))

                parent.left = node_map[left_key]
                parent.right = node_map[right_key]

                node_map[left_key].parent = parent
                node_map[right_key].parent = parent

                node_map[fusion_key] = parent

            root = node_map[fusion_key]

        return root

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self.tree: TreeNode | None = None
        self.fusion_history: List = []  # Para depuración y construcción del árbol

    # ====================== EVALUACIÓN DE k-PARTICIÓN ======================
    def _evaluate_k_partition(
        self, blocks: List[List[Tuple[int, int]]]
    ) -> Tuple[float, np.ndarray]:
        """Evaluación segura - filtra vacíos y usa particionar_multi_k"""
        # Filtrar bloques vacíos
        blocks = [b for b in blocks if len(b) > 0]
        if len(blocks) < 2:
            return INFTY_POS, None

        particion_alcance = [[idx for t, idx in b if t == EFFECT] for b in blocks]
        particion_mecanismo = [[idx for t, idx in b if t == ACTUAL] for b in blocks]

        # ── DEBUG TEMPORAL ──────────────────────────────────────────
        # print("hola")
        # print(f"\n[DEBUG _evaluate_k_partition]")
        # print(f"  p_v esperado len : {len(self.sia_dists_marginales)}")
        # print(f"  subsistema indices: {self.sia_subsistema.indices_ncubos}")
        # print(f"  subsistema dims   : {self.sia_subsistema.dims_ncubos}")
        # for i, (alc, mec) in enumerate(zip(particion_alcance, particion_mecanismo)):
        #     print(f"  Bloque {i}: alcance={alc}  mecanismo={mec}")

        try:
            sistema_part = self.sia_subsistema.particionar_multi_k(
                particion_alcance, particion_mecanismo
            )
            hat_p = sistema_part.distribucion_marginal()
            # print(f"  hat_p = {hat_p}")
            # print(f"  ncubos resultantes = {sistema_part.indices_ncubos}")
            p_v = self.sia_dists_marginales

            if len(hat_p) != len(p_v):
                print(f"Dimension mismatch: hat_p={len(hat_p)} vs p_v={len(p_v)}")
                return INFTY_POS, None

            loss = emd_efecto(hat_p, p_v)
            return loss, hat_p
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return INFTY_POS, None

    def _generate_k_cuts(self, k: int):
        """Genera cortes evitando bloques vacíos"""
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
                # Validación fuerte contra bloques inútiles
                good = all(
                    any(t == EFFECT for t, _ in b) or any(t == ACTUAL for t, _ in b)
                    for b in valid_blocks
                )
                if good:
                    signature = tuple(sorted(tuple(b) for b in valid_blocks))
                    if signature not in seen:
                        seen.add(signature)
                        candidates.append(valid_blocks)

        # print(f"→ Candidatos válidos del árbol: {len(candidates)}")

        # # En _generate_k_cuts, antes del return:
        # print(f"[DEBUG] Nodos internos del árbol: {len(internal)}")
        # print(
        #     f"[DEBUG] Combinaciones posibles C({len(internal)},{k-1}): "
        #     f"{len(list(combinations(range(len(internal)), k-1)))}"
        # )
        # print(f"[DEBUG] Candidatos generados: {len(candidates)}")

        # Fallback robusto
        if len(candidates) < 5:
            print("Usando fallback balanceado")
            all_leaves = sorted(self.tree.get_all_leaves(), key=lambda x: (x[0], x[1]))
            n = len(all_leaves)
            sizes = [max(1, n // k)] * k
            for i in range(n % k):
                sizes[i] += 1
            start = 0
            fallback = []
            for size in sizes:
                fallback.append(all_leaves[start : start + size])
                start += size
            if len(fallback) == k:
                candidates.append(fallback)

        return candidates[:25]  # Limitamos para no explotar tiempo

    # ====================== ÁRBOL Y CORTES ======================
    def _run_with_tree(self, vertices):
        self.fusion_history.clear()
        self.algorithm(vertices)
        self.tree = self._build_tree_from_fusions()
        return self.tree

    def _get_all_internal_edges(self, node: TreeNode, edges: List = None) -> List:
        """Recolecta aristas internas del árbol para generar cortes."""
        if edges is None:
            edges = []
        if node.left and node.right:
            edges.append((node, node.left, node.right))
            self._get_all_internal_edges(node.left, edges)
            self._get_all_internal_edges(node.right, edges)
        return edges

    def _collect_internal_nodes(self, node, res):
        if node is None:
            return

        if node.left and node.right:
            res.append(node)

        self._collect_internal_nodes(node.left, res)
        self._collect_internal_nodes(node.right, res)

    def _is_ancestor(self, ancestor, node):
        current = node.parent

        while current is not None:

            if current is ancestor:
                return True

            current = current.parent

        return False

        # Fallback fuerte si no hay suficientes
        if len(candidates) < 5:
            print("Usando fallback balanceado")
            all_leaves = sorted(self.tree.get_all_leaves(), key=lambda x: (x[0], x[1]))
            n = len(all_leaves)
            sizes = [max(1, n // k)] * k
            for i in range(n % k):
                sizes[i] += 1
            start = 0
            fallback = []
            for size in sizes:
                fallback.append(all_leaves[start : start + size])
                start += size
            candidates.append(fallback)

        return candidates[:30]

    @staticmethod
    def fmt_particion_multi_k(
        particion_alcance: list[list[int]],
        particion_mecanismo: list[list[int]],
    ) -> str:

        linea_superior = ""
        linea_inferior = ""

        for alcance, mecanismo in zip(
            particion_alcance,
            particion_mecanismo,
        ):

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
    def aplicar_estrategia_k(self, estado_inicial, condicion, alcance, mecanismo, k=3):
        inicio = time.perf_counter()
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        # Caso degenerado
        if np.allclose(self.sia_dists_marginales, 0.0):
            return self._solucion_trivial(k, inicio)

        presente = [(ACTUAL, idx) for idx in self.sia_subsistema.dims_ncubos]
        futuro = [(EFFECT, idx) for idx in self.sia_subsistema.indices_ncubos]

        # ── Generar pool de candidatos con múltiples estrategias ──────────────────
        pool = self._generar_pool_candidatos(presente, futuro, k)

        # ── Evaluar todos y tomar el mejor ────────────────────────────────────────
        best_blocks, best_loss, best_dist = self._find_best_partition(pool)

        if best_blocks is None:
            best_blocks = pool[0]
            best_loss = INFTY_POS
            best_dist = self.sia_dists_marginales

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

    # ── Helpers nuevos ─────────────────────────────────────────────────────────────

    def _bloques_son_validos(self, blocks: list, k: int) -> bool:
        """
        Retorna True solo si:
        - Hay exactamente k bloques
        - Cada bloque tiene al menos 1 ACTUAL  (mecanismo no vacío)
        - Cada bloque tiene al menos 1 EFFECT  (alcance no vacío)
        """
        if len(blocks) != k:
            return False
        for b in blocks:
            tiene_actual = any(t == ACTUAL for t, _ in b)
            tiene_effect = any(t == EFFECT for t, _ in b)
            if not tiene_actual or not tiene_effect:
                return False
        return True

    def _generar_particion_balanceada(self, vertices: list, k: int) -> list[list]:
        """
        Fallback garantizado: distribuye vértices en k bloques
        asegurando que cada bloque reciba al menos un ACTUAL y un EFFECT.
        Estrategia: round-robin sobre los vértices ya intercalados.
        """
        bloques = [[] for _ in range(k)]
        for i, v in enumerate(vertices):
            bloques[i % k].append(v)

        # Verificar y reparar si algún bloque quedó sin ACTUAL o EFFECT
        for i, bloque in enumerate(bloques):
            if not any(t == ACTUAL for t, _ in bloque):
                # Robar un ACTUAL del bloque más grande que tenga ≥2
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

    def _fallback_solution(self, k: int, inicio: float) -> Solution:
        """Solución de emergencia cuando el árbol no se puede construir."""
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
            estrategia=f"KQNodes-Fallback(k={k})",
            perdida=INFTY_POS,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.sia_dists_marginales,
            particion=fmt,
            tiempo_total=time.perf_counter() - inicio,
        )

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def algorithm(self, vertices: list[tuple[int, int]]):
        """
        Implementa el algoritmo Q para encontrar la partición óptima de un sistema que minimiza la pérdida de información, basándose en principios de submodularidad dentro de la teoría de lainformación.

        El algoritmo opera sobre un conjunto de vértices que representan nodos en diferentes tiempos del sistema (presente y futuro). La idea fundamental es construir incrementalmente grupos de nodos que, cuando se particionan, producen la menor pérdida posible de información en el sistema.

        Proceso Principal:
        -----------------
        El algoritmo comienza estableciendo dos conjuntos fundamentales: omega (W) y delta.
        Omega siempre inicia con el primer vértice del sistema, mientras que delta contiene todos los vértices restantes. Esta decisión no es arbitraria - al comenzar con un
        solo elemento en omega, podemos construir grupos de manera incremental evaluando cómo cada adición afecta la pérdida de información.

        La ejecución se desarrolla en fases, ciclos e iteraciones, donde cada fase representa un nivel diferente y conlleva a la formación de una partición candidata, cada ciclo representa un incremento de elementos al conjunto W y cada iteración determina al final cuál es el mejor elemento/cambio/delta para añadir en W.
        Fase >> Ciclo >> Iteración.

        1. Formación Incremental de Grupos:
        El algoritmo mantiene un conjunto omega que crece gradualmente en cada j-iteración. En cada paso, evalúa todos los deltas restantes para encontrar cuál, al unirse con omega produce la menor pérdida de información. Este proceso utiliza la función submodular para calcular la diferencia entre la EMD (Earth Mover's Distance) de la combinación y la EMD individual del delta evaluado.

        2. Evaluación de deltas:
        Para cada delta candidato el algoritmo:
        - Calcula su EMD individual si no está en memoria.
        - Calcula la EMD de su combinación con el conjunto omega actual
        - Determina la diferencia entre estas EMDs (el "costo" de la combinación)
        El delta que produce el menor costo se selecciona y se añade a omega.

        3. Formación de Nuevos Grupos:
        Al final de cada fase cuando omega crezca lo suficiente, el algoritmo:
        - Toma los últimos elementos de omega y delta (par candidato).
        - Los combina en un nuevo grupo
        - Actualiza la lista de vértices para la siguiente fase
        Este proceso de agrupamiento permite que el algoritmo construya particiones
        cada vez más complejas y reutilice estos "pares candidatos" para particiones en conjunto.

        Optimización y Memoria:
        ----------------------
        El algoritmo utiliza dos estructuras de memoria clave:
        - individual_memory: Almacena las EMDs y distribuciones de nodos individuales, evitando recálculos muy costosos.
        - partition_memory: Guarda las EMDs y distribuciones de las particiones completas, permitiendo comparar diferentes combinaciones de grupos teniendo en cuenta que su valor real está asociado al valor individual de su formación delta.

        La memoización es relevante puesto muchos cálculos de EMD son computacionalmente costosos y se repiten durante la ejecución del algoritmo.

        Resultado:
        ---------------
        Al terminar todas las fases, el algoritmo selecciona la partición que produjo la menor EMD global, representando la división del sistema que mejor preserva su información causal.

        Args:
            vertices (list[tuple[int, int]]): Lista de vértices donde cada uno es una
                tupla (tiempo, índice). tiempo=0 para presente (t_0), tiempo=1 para futuro (t_1).

        Returns:
            tuple[float, tuple[tuple[int, int], ...]]: El valor de pérdida en la primera posición, asociado con la partición óptima encontrada, identificada por la clave en partition_memory que produce la menor EMD.
        """
        indice_emd = INT_ZERO

        for i in range(len(vertices) - 1):
            # self.logger.debug(f"total: {len(vertices) - i}")
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for j in range(len(deltas_ciclo) - 1):
                # self.logger.critic(f"   {j=}")
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
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_marginal_delta,
                            )
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = k
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta
                # self.logger.critic(f"       [k]: {indice_mip}")

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
            # FUSION
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

            self.fusion_history.append(
                (
                    left_group,
                    right_group,
                    fusion_group,
                )
            )

            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)

            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k][indice_emd],
        )

    def _find_best_partition(self, candidates):
        print("CANDIDATOS", len(candidates))
        """Evalúa todos los candidatos y retorna el mejor."""
        best_loss = INFTY_POS
        best_blocks = None
        best_dist = None

        # En _find_best_partition, antes del loop:
        # print(f"[DEBUG] Evaluando {len(candidates)} candidatos")
        # for i, c in enumerate(candidates):
        #     alc = [[idx for t, idx in b if t == EFFECT] for b in c]
        #     mec = [[idx for t, idx in b if t == ACTUAL] for b in c]
        #     print(f"  Candidato {i}: alc={alc} mec={mec}")

        for candidate in candidates:
            loss, dist = self._evaluate_k_partition(candidate)
            if loss < best_loss:
                best_loss = loss
                best_blocks = candidate
                best_dist = dist

        return best_blocks, best_loss, best_dist

    # nuevo

    def _generar_pool_candidatos(self, presente, futuro, k):
        """
        Genera candidatos explorando particiones independientes de
        alcance y mecanismo, como hace fuerza bruta pero de forma dirigida.
        """
        indices_futuros = sorted(set(idx for _, idx in futuro))
        indices_presentes = sorted(set(idx for _, idx in presente))

        map_effect = {idx: (EFFECT, idx) for idx in indices_futuros}
        map_actual = {idx: (ACTUAL, idx) for idx in indices_presentes}

        pool = []
        seen = set()

        def registrar(part_alcance, part_mec):
            """Convierte dos particiones independientes en bloques de vértices."""
            # Normalizar a exactamente k grupos no vacíos en cada lado
            if len(part_alcance) != k or len(part_mec) != k:
                return False

            # Emparejar bloque i de alcance con bloque i de mecanismo
            blocks = []
            for alc, mec in zip(part_alcance, part_mec):
                bloque = []
                for idx in alc:
                    if idx in map_effect:
                        bloque.append(map_effect[idx])
                for idx in mec:
                    if idx in map_actual:
                        bloque.append(map_actual[idx])
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

        # ── Estrategia 1: Q independiente sobre futuros ───────────────────────────
        parts_futuros = self._q_particiones(indices_futuros, k)
        # ── Estrategia 2: Q independiente sobre presentes ────────────────────────
        parts_presentes = self._q_particiones(indices_presentes, k)

        # ── Combinar: cada partición de futuros × cada partición de presentes ─────
        # Esto replica la estructura de generar_particiones_multi_k pero dirigido
        for p_alc in parts_futuros:
            for p_mec in parts_presentes:
                registrar(p_alc, p_mec)

        print(f"→ Candidatos tras Q×Q: {len(pool)}")

        # ── Estrategia 3: particiones estructurales de futuros ────────────────────
        n_f = len(indices_futuros)
        n_p = len(indices_presentes)

        parts_struct_f = self._particiones_estructurales(indices_futuros, k)
        parts_struct_p = self._particiones_estructurales(indices_presentes, k)

        # Combinar estructurales con Q y entre sí
        for p_alc in parts_futuros + parts_struct_f:
            for p_mec in parts_presentes + parts_struct_p:
                registrar(p_alc, p_mec)

        print(f"→ Total candidatos en pool: {len(pool)}")

        # Garantizar al menos 1 candidato válido
        if not pool:
            grupos_f = [indices_futuros[i::k] for i in range(k)]
            grupos_p = [indices_presentes[i::k] for i in range(k)]
            registrar(grupos_f, grupos_p)

        return pool[:50]  # tope razonable para mantener tiempo controlado

    def _q_particiones(self, indices: list, k: int) -> list[list[list[int]]]:
        """
        Corre el algoritmo Q sobre una lista de índices y extrae
        múltiples particiones en k grupos desde el árbol resultante.
        Retorna lista de particiones, cada una es lista de k grupos de índices.
        """
        if len(indices) < k:
            return []

        # Representar como vértices ACTUAL para correr el algoritmo Q
        vertices = [(ACTUAL, idx) for idx in indices]

        hist_backup = self.fusion_history.copy()
        mem_backup = dict(self.memoria_grupo_candidato)
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
            seen_part = set()

            for cut_combo in combinations(internal, k - 1):
                cut_set = set(cut_combo)
                grupos = []

                def dfs(node):
                    if node is None:
                        return
                    if node in cut_set or node.is_leaf:
                        leaves = node.get_all_leaves()
                        idxs = [idx for _, idx in leaves]
                        if idxs:
                            grupos.append(sorted(idxs))
                        return
                    dfs(node.left)
                    dfs(node.right)

                dfs(arbol)

                if len(grupos) == k and all(len(g) > 0 for g in grupos):
                    sig = tuple(sorted(tuple(g) for g in grupos))
                    if sig not in seen_part:
                        seen_part.add(sig)
                        particiones.append(grupos)

                if len(particiones) >= 10:  # tope por árbol
                    break

            return particiones

        except Exception as e:
            print(f"[_q_particiones] Error: {e}")
            return []
        finally:
            self.fusion_history = hist_backup
            self.memoria_grupo_candidato = mem_backup

    def _particiones_estructurales(
        self, indices: list, k: int
    ) -> list[list[list[int]]]:
        """
        Genera particiones estructurales simples del conjunto de índices:
        bloques contiguos y round-robin con distintos offsets.
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
                particiones.append(grupos)

        # Bloques contiguos con distintos offsets
        tam = max(1, n // k)
        for offset in range(k):
            rotado = indices[offset:] + indices[:offset]
            grupos = [sorted(rotado[i * tam : (i + 1) * tam]) for i in range(k)]
            sobrantes = rotado[k * tam :]
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

        # Por valor de distribución marginal (si disponible)
        if hasattr(self, "sia_dists_marginales") and len(
            self.sia_dists_marginales
        ) >= max(indices, default=0):
            indices_por_dist = sorted(
                indices,
                key=lambda i: (
                    self.sia_dists_marginales[i]
                    if i < len(self.sia_dists_marginales)
                    else 0
                ),
            )
            grupos = [[] for _ in range(k)]
            for i, idx in enumerate(indices_por_dist):
                grupos[i % k].append(idx)
            agregar(grupos)

        return particiones
