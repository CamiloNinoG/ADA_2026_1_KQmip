import time
from typing import List, Tuple
import numpy as np
from itertools import combinations
from src.strategies.kqnodes.tree_node import TreeNode
from src.strategies.q_nodes import QNodes
from src.models.core.solution import Solution
from src.constants.models import QNODES_LABEL, QNODES_ANALYSIS_TAG
from src.constants.base import ACTUAL, EFFECT, INT_ZERO, INFTY_POS, LAST_IDX, TYPE_TAG
from src.funcs.iit import LOWER_ABECEDARY, emd_efecto, ABECEDARY
from src.funcs.format import fmt_particion_multi_k
from src.middlewares.profile import gestor_perfilado, profile


# MEJOR PARTICON
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

    # ====================== EVALUACIÓN ======================

    def _evaluate_k_partition(
        self, blocks: List[List[Tuple[int, int]]]
    ) -> Tuple[float, np.ndarray]:
        """
        Evalúa una k-partición usando _bipartir_y_emd por bloque.
        No crea copias de tensores completos → sin OOM.
        """
        blocks = [b for b in blocks if len(b) > 0]
        if len(blocks) < 2:
            return INFTY_POS, None

        indices_sistema = list(self.sia_subsistema.indices_ncubos)
        hat_p = np.zeros(len(indices_sistema), dtype=np.float32)

        for bloque in blocks:
            alc = sorted([idx for t, idx in bloque if t == EFFECT])
            mec = sorted([idx for t, idx in bloque if t == ACTUAL])

            # Restricción IIT: el alcance (futuro) NO puede estar vacío en ningún bloque.
            # El mecanismo (presente) SÍ puede estar vacío.
            if not alc:
                return INFTY_POS, None

            try:
                # Bipartir y obtener dist (funciona tanto para mec normal como mec vacio)
                emd_bloque, dist_bloque = self._bipartir_y_emd(alc, mec, solo_emd=False)
            except Exception as e:
                print(f"[_evaluate_k_partition] Error bloque alc={alc} mec={mec}: {e}")
                return INFTY_POS, None

            # Colocar distribución del bloque en las posiciones correctas de hat_p usando el índice global
            for idx in alc:
                if idx in indices_sistema:
                    pos = indices_sistema.index(idx)
                    hat_p[pos] = dist_bloque[pos]

        if len(hat_p) != len(self.sia_dists_marginales):
            return INFTY_POS, None

        loss = emd_efecto(hat_p, self.sia_dists_marginales)
        return loss, hat_p

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

    # ====================== GENERACIÓN DE CANDIDATOS ======================

    def _q_particiones(self, indices: list, k: int) -> list[list[list[int]]]:
        """
        Corre el algoritmo Q sobre una lista de índices y extrae
        múltiples particiones en k grupos desde el árbol resultante
        usando división voraz y muestreo probabilístico sobre el árbol.
        """
        if len(indices) < k:
            return []

        vertices = [(ACTUAL, idx) for idx in indices]

        # Backup del estado para no contaminarlo
        hist_backup = self.fusion_history.copy()
        mem_backup = dict(self.memoria_grupo_candidato)
        self.fusion_history.clear()
        self.memoria_grupo_candidato.clear()

        try:
            self.algorithm(vertices)

            arbol = self._build_tree_from_fusions()
            if arbol is None:
                return []

            particiones = []
            seen_part = set()

            # 1. Candidato Determinista: Dividir siempre el nodo con más hojas descendentes
            current_nodes = [arbol]
            while len(current_nodes) < k:
                non_leaves = [n for n in current_nodes if not n.is_leaf]
                if not non_leaves:
                    break
                # Ordenar por el número de hojas de mayor a menor
                non_leaves.sort(key=lambda n: len(n.get_all_leaves()), reverse=True)
                node_to_split = non_leaves[0]
                current_nodes.remove(node_to_split)
                current_nodes.append(node_to_split.left)
                current_nodes.append(node_to_split.right)

            grupos = []
            for n in current_nodes:
                leaves = n.get_all_leaves()
                idxs = [idx for _, idx in leaves]
                if idxs:
                    grupos.append(sorted(idxs))
            if len(grupos) == k and all(len(g) > 0 for g in grupos):
                sig = tuple(sorted(tuple(g) for g in grupos))
                seen_part.add(sig)
                particiones.append(grupos)

            # 2. Candidatos Aleatorios Ponderados: Obtener diversidad
            np.random.seed(42)  # Semilla fija para consistencia en pruebas
            intentos = 0
            while len(particiones) < 15 and intentos < 100:
                intentos += 1
                current_nodes = [arbol]
                while len(current_nodes) < k:
                    non_leaves = [n for n in current_nodes if not n.is_leaf]
                    if not non_leaves:
                        break
                    # Ponderación proporcional al tamaño al cuadrado
                    weights = [len(n.get_all_leaves()) ** 2 for n in non_leaves]
                    total_w = sum(weights)
                    probs = [w / total_w for w in weights]
                    node_to_split = np.random.choice(non_leaves, p=probs)

                    current_nodes.remove(node_to_split)
                    current_nodes.append(node_to_split.left)
                    current_nodes.append(node_to_split.right)

                grupos = []
                for n in current_nodes:
                    leaves = n.get_all_leaves()
                    idxs = [idx for _, idx in leaves]
                    if idxs:
                        grupos.append(sorted(idxs))
                if len(grupos) == k and all(len(g) > 0 for g in grupos):
                    sig = tuple(sorted(tuple(g) for g in grupos))
                    if sig not in seen_part:
                        seen_part.add(sig)
                        particiones.append(grupos)

            return particiones

        except Exception as e:
            print(f"[_q_particiones] Error: {e}")
            return []
        finally:
            # Siempre restaurar el estado original
            self.fusion_history = hist_backup
            self.memoria_grupo_candidato = mem_backup

    def _particiones_estructurales(
        self, indices: list, k: int
    ) -> list[list[list[int]]]:
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

        # Por valor de distribución marginal
        n_dist = len(self.sia_dists_marginales)
        if indices and max(indices, default=0) < n_dist:
            indices_por_dist = sorted(
                indices, key=lambda i: self.sia_dists_marginales[i]
            )
            grupos = [[] for _ in range(k)]
            for i, idx in enumerate(indices_por_dist):
                grupos[i % k].append(idx)
            agregar(grupos)

        return particiones

    def _generar_pool_candidatos(self, presente, futuro, k):
        indices_futuros = sorted(set(idx for _, idx in futuro))
        indices_presentes = sorted(set(idx for _, idx in presente))
        n = len(indices_futuros)

        map_effect = {idx: (EFFECT, idx) for idx in indices_futuros}
        map_actual = {idx: (ACTUAL, idx) for idx in indices_presentes}

        pool = []
        seen = set()

        def _bloques_validos_con_vacio(blocks):
            """Validación que permite mecanismo vacío (∅) en cualquier bloque."""
            if len(blocks) != k:
                return False
            for b in blocks:
                if not any(t == EFFECT for t, _ in b):
                    return False  # alcance vacío nunca es válido
            return True

        def registrar(part_alcance, part_mec, permitir_vacio_mec=False):
            if len(part_alcance) != k or len(part_mec) != k:
                return False
            blocks = []
            for alc, mec in zip(part_alcance, part_mec):
                bloque = [map_effect[idx] for idx in alc if idx in map_effect]
                bloque += [map_actual[idx] for idx in mec if idx in map_actual]
                if bloque:
                    blocks.append(bloque)
            # Usar validación permisiva si se indica
            valido = _bloques_validos_con_vacio(blocks) if permitir_vacio_mec else self._bloques_son_validos(blocks, k)
            if not valido:
                return False
            sig = tuple(sorted(tuple(sorted(b)) for b in blocks))
            if sig in seen:
                return False
            seen.add(sig)
            pool.append(blocks)
            return True

        # Q solo para sistemas pequeños donde aporta sin OOM
        parts_futuros = []
        parts_presentes = []
        if n < 23:
            parts_futuros = self._q_particiones(indices_futuros, k)
            parts_presentes = self._q_particiones(indices_presentes, k)
            for p_alc in parts_futuros:
                for p_mec in parts_presentes:
                    registrar(p_alc, p_mec)

        print(f"→ Candidatos tras Q×Q: {len(pool)}")

        # Estructurales siempre — baratas y sin tensores grandes
        parts_struct_f = self._particiones_estructurales(indices_futuros, k)
        parts_struct_p = self._particiones_estructurales(indices_presentes, k)

        for p_alc in parts_futuros + parts_struct_f:
            for p_mec in parts_presentes + parts_struct_p:
                registrar(p_alc, p_mec)

        # ─── CANDIDATOS CON MECANISMO VACÍO ───────────────────────────────────
        # Añadir directamente particiones donde 1 bloque tiene mec=∅.
        # Esto es crítico para K=2: (A/∅ | resto/todo) es el óptimo en muchos sistemas.
        # Para K>2 la misma idea aplica. Coste: n_futuros evaluaciones, barato.
        if n < 23:  # Solo para tamaños manejables
            # Para cada elemento individual del futuro, crear (elem/∅ | resto/todo)
            for i, idx_solo in enumerate(indices_futuros):
                resto_f = [x for x in indices_futuros if x != idx_solo]
                if not resto_f:
                    continue
                # Bloque 1: solo ese futuro, mec vacío
                # Bloque 2: el resto del futuro + todo el presente
                p_alc = [[idx_solo]] + [resto_f] + [[] for _ in range(k - 2)]
                p_mec = [[]] + [indices_presentes] + [[] for _ in range(k - 2)]
                if k == 2:
                    registrar([p_alc[0], p_alc[1]], [p_mec[0], p_mec[1]], permitir_vacio_mec=True)
                    # Invertido: el bloque grande con mec vacío
                    registrar([p_alc[1], p_alc[0]], [p_mec[1], p_mec[0]], permitir_vacio_mec=True)
            # Para K>2: solo un bloque por vez tiene mec vacío
            if k > 2:
                parts_todos_f = parts_futuros + parts_struct_f
                for p_alc in parts_todos_f[:5]:  # limitar para no explotar pool
                    for j in range(k):
                        p_mec_vacio = []
                        for ki in range(k):
                            p_mec_vacio.append([] if ki == j else indices_presentes)
                        registrar(p_alc, p_mec_vacio, permitir_vacio_mec=True)

        print(f"→ Total candidatos en pool (con ∅): {len(pool)}")

        if not pool:
            grupos_f = [indices_futuros[i::k] for i in range(k)]
            grupos_p = [indices_presentes[i::k] for i in range(k)]
            registrar(grupos_f, grupos_p)

        # Tope adaptativo según tamaño del sistema
        if n >= 25:
            tope = 5
        elif n >= 22:
            tope = 10
        elif n >= 20:
            tope = 15
        elif n >= 15:
            tope = 35
        else:
            tope = 80  # Sistemas pequeños: pool completo
        return pool[:tope]

    # ====================== HELPERS ======================

    def _bloques_son_validos(self, blocks: list, k: int, allow_empty_mec: bool = False) -> bool:
        """
        Verifica que haya exactamente k bloques. Reglas:
          - Cada bloque debe tener ≥1 variable EFFECT (alc ≠ ∅): sin futuro, el bloque es inválido.
          - El mecanismo puede ser vacío (mec = ∅) solo si allow_empty_mec es True.
        """
        if len(blocks) != k:
            return False
        for b in blocks:
            if not any(t == EFFECT for t, _ in b):
                return False
            if not allow_empty_mec and not any(t == ACTUAL for t, _ in b):
                return False
        return True

    def _generar_particion_balanceada(self, vertices: list, k: int) -> list[list]:
        """Round-robin garantizando al menos 1 EFFECT en cada bloque (ACTUAL puede ser vacío)."""
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

    def _refinar_busqueda_local(
        self, blocks: List[List[Tuple[int, int]]], k: int, allow_empty_mec: bool = False
    ) -> Tuple[List[List[Tuple[int, int]]], float, np.ndarray]:
        """
        Búsqueda local (Hill Climbing) con dos tipos de movimiento:
        1. Mover un elemento individual entre bloques.
        2. [NUEVO] Vaciar el mecanismo completo de un bloque ("salto de vaciado").
           Permite escapar mínimos locales donde la ruta unitaria atraviesa
           una montaña de pérdida pero el destino (mec=∅) es mejor.
        """
        import traceback

        try:
            best_blocks = [list(b) for b in blocks]
            best_loss, best_dist = self._evaluate_k_partition(best_blocks)

            mejorado = True
            paso = 0
            max_pasos = 50

            while mejorado and paso < max_pasos:
                mejorado = False
                paso += 1

                # ─── TIPO 1: Mover elemento individual ──────────────────────
                elementos = []
                for b in best_blocks:
                    elementos.extend(b)

                for elem in elementos:
                    bloque_origen_idx = -1
                    for idx, b in enumerate(best_blocks):
                        if any(elem[0] == x[0] and elem[1] == x[1] for x in b):
                            bloque_origen_idx = idx
                            break

                    if bloque_origen_idx == -1:
                        continue

                    if len(best_blocks[bloque_origen_idx]) <= 1:
                        continue

                    t_elem, _ = elem
                    if t_elem == EFFECT:
                        tiene_otro_effect = any(
                            t == EFFECT and not (t == elem[0] and val == elem[1])
                            for t, val in best_blocks[bloque_origen_idx]
                        )
                        if not tiene_otro_effect:
                            continue
                    elif t_elem == ACTUAL and not allow_empty_mec:
                        tiene_otro_actual = any(
                            t == ACTUAL and not (t == elem[0] and val == elem[1])
                            for t, val in best_blocks[bloque_origen_idx]
                        )
                        if not tiene_otro_actual:
                            continue

                    for bloque_destino_idx in range(k):
                        if bloque_destino_idx == bloque_origen_idx:
                            continue

                        temp_blocks = [list(b) for b in best_blocks]
                        for x in temp_blocks[bloque_origen_idx]:
                            if x[0] == elem[0] and x[1] == elem[1]:
                                temp_blocks[bloque_origen_idx].remove(x)
                                break
                        temp_blocks[bloque_destino_idx].append(elem)

                        loss, dist = self._evaluate_k_partition(temp_blocks)
                        if loss < best_loss:
                            best_loss = loss
                            best_dist = dist
                            best_blocks = temp_blocks
                            mejorado = True
                            break

                    if mejorado:
                        break

                if mejorado:
                    continue

                # ─── TIPO 2: Salto de vaciado (solo si allow_empty_mec) ─────
                # Evalúa vaciar el mecanismo completo de un bloque de golpe.
                # Necesario para escapar mínimos locales: el camino unitario
                # empeora la pérdida pero el destino (mec=∅) es mejor.
                if allow_empty_mec:
                    for i_bloque in range(k):
                        actuales_en_bloque = [
                            e for e in best_blocks[i_bloque] if e[0] == ACTUAL
                        ]
                        if not actuales_en_bloque:
                            continue  # ya está vacío

                        # Construir bloque sin ningún elemento ACTUAL
                        temp_blocks = [list(b) for b in best_blocks]
                        temp_blocks[i_bloque] = [
                            e for e in temp_blocks[i_bloque] if e[0] == EFFECT
                        ]
                        if not temp_blocks[i_bloque]:  # quedaría vacío → inválido
                            continue

                        loss, dist = self._evaluate_k_partition(temp_blocks)
                        if loss < best_loss:
                            best_loss = loss
                            best_dist = dist
                            best_blocks = temp_blocks
                            mejorado = True
                            break

            return best_blocks, best_loss, best_dist
        except Exception as e:
            traceback.print_exc()
            raise e

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

        pool = self._generar_pool_candidatos(presente, futuro, k)
        best_blocks, best_loss, best_dist = self._find_best_partition(pool)

        if best_blocks is None:
            best_blocks = pool[0]
            best_loss = INFTY_POS
            best_dist = self.sia_dists_marginales
        else:
            # Aplicar refinamiento por búsqueda local (Hill Climbing)
            n_nodos = len(self.sia_subsistema.indices_ncubos)
            if best_blocks is not None and n_nodos < 20:
                print(f"   [HC] Pérdida pool antes: {best_loss:.6f}")

                # Fase 1 – Restringida: sin mecanismos vacíos (ancla estructural)
                b1, l1, d1 = self._refinar_busqueda_local(best_blocks, k, allow_empty_mec=False)

                # Fase 2 – Relajada: con saltos de vaciado desde mismo inicio
                b2, l2, d2 = self._refinar_busqueda_local(best_blocks, k, allow_empty_mec=True)

                # Fase 3 – Relajada desde el mejor resultado de Fase 1
                # (permite escapar mínimos locales distintos del inicio)
                b3, l3, d3 = self._refinar_busqueda_local(b1, k, allow_empty_mec=True)

                mejor = min([(l1, b1, d1), (l2, b2, d2), (l3, b3, d3)], key=lambda x: x[0])
                best_loss, best_blocks, best_dist = mejor[0], mejor[1], mejor[2]
                print(f"   [HC] Pérdida final: {best_loss:.6f}")
            print(
                f"   [Refinamiento] Pérdida final: {best_loss:.6f}"
            )

        particion_alcance = [[idx for t, idx in b if t == EFFECT] for b in best_blocks]
        particion_mecanismo = [
            [idx for t, idx in b if t == ACTUAL] for b in best_blocks
        ]
        fmt = self.fmt_particion_multi_k(particion_alcance, particion_mecanismo)

        # print("\n=== MEMORIAS ===")
        # print("memoria_delta:", len(self.memoria_delta))
        # print("memoria_grupo_candidato:", len(self.memoria_grupo_candidato))
        # print("_bipartir_emd_cache:", len(self._bipartir_emd_cache))
        # print("================\n")

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
                            self.memoria_grupo_candidato[clave] = (
                                emd_delta,
                                dist_marginal_delta,
                            )
                            return clave

                        emd_local = emd_iteracion
                        indice_mip = k
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

        print(
            "algorithm" "delta:",
            len(self.memoria_delta),
            "grupo:",
            len(self.memoria_grupo_candidato),
            "bip:",
            len(self._bipartir_emd_cache),
        )

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k][indice_emd],
        )
