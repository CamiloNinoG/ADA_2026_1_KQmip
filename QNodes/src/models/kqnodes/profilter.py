"""
KQNodes Profiler — Pega este archivo en tu proyecto y llama:
    from kqnodes_profiler import KQNodesProfiler
    KQNodesProfiler.wrap(tu_instancia_kqnodes)

Luego ejecuta normalmente aplicar_estrategia_k y al final imprime:
    KQNodesProfiler.report()
"""

import time
import functools
from collections import defaultdict
from typing import Callable
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Motor de medición
# ──────────────────────────────────────────────────────────────────────────────

class _Timer:
    def __init__(self):
        self.calls: int = 0
        self.total_ms: float = 0.0
        self.min_ms: float = float("inf")
        self.max_ms: float = 0.0
        self.samples: list[float] = []   # últimas 200 muestras para percentiles

    def record(self, elapsed_ms: float):
        self.calls += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)
        if len(self.samples) < 200:
            self.samples.append(elapsed_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]


class KQNodesProfiler:
    """
    Profiler no-intrusivo para KQNodes.
    Instrumenta métodos clave en tiempo de ejecución sin tocar el código original.
    """

    _timers: dict[str, _Timer] = defaultdict(_Timer)
    _cache_hits: dict[str, int] = defaultdict(int)
    _cache_misses: dict[str, int] = defaultdict(int)
    _candidates_counts: list[int] = []
    _k_values: list[int] = []
    _wall_start: float = 0.0

    # Métodos a instrumentar automáticamente
    _TARGET_METHODS = [
        "algorithm",
        "_build_tree_from_fusions",
        "_generate_k_cuts",
        "_evaluate_k_partition",
        "_find_best_partition",
        "_bloques_son_validos",
        "_generar_particion_balanceada",
        "funcion_submodular",
        "aplicar_estrategia_k",
    ]

    @classmethod
    def wrap(cls, instance) -> None:
        """
        Instrumenta todos los métodos objetivo en la instancia dada.
        Uso: KQNodesProfiler.wrap(mi_kqnodes)
        """
        cls._wall_start = time.perf_counter()
        cls._timers.clear()
        cls._cache_hits.clear()
        cls._cache_misses.clear()
        cls._candidates_counts.clear()
        cls._k_values.clear()

        for method_name in cls._TARGET_METHODS:
            if hasattr(instance, method_name):
                original = getattr(instance, method_name)
                wrapped = cls._make_wrapper(method_name, original)
                setattr(instance, method_name, wrapped)

        # Parche especial: trackear cache hits en funcion_submodular
        cls._patch_cache_tracking(instance)

        print("✅ KQNodesProfiler activo — métodos instrumentados:")
        for m in cls._TARGET_METHODS:
            status = "✓" if hasattr(instance, m) else "✗ (no encontrado)"
            print(f"   {status} {m}")

    @classmethod
    def _make_wrapper(cls, name: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            cls._timers[name].record(elapsed)

            # Captura extra según el método
            if name == "_generate_k_cuts" and result is not None:
                cls._candidates_counts.append(len(result))
            if name == "aplicar_estrategia_k":
                k = kwargs.get("k", args[4] if len(args) > 4 else "?")
                cls._k_values.append(k)

            return result
        return wrapper

    @classmethod
    def _patch_cache_tracking(cls, instance) -> None:
        """Envuelve memoria_delta para contar hits/misses."""
        original_dict = instance.memoria_delta if hasattr(instance, "memoria_delta") else None
        if original_dict is None:
            return

        profiler_ref = cls

        class TrackingDict(dict):
            def __contains__(self, key):
                result = super().__contains__(key)
                if result:
                    profiler_ref._cache_hits["memoria_delta"] += 1
                else:
                    profiler_ref._cache_misses["memoria_delta"] += 1
                return result

        tracked = TrackingDict(original_dict)
        instance.memoria_delta = tracked

    @classmethod
    def report(cls) -> None:
        """Imprime el reporte completo de rendimiento."""
        wall = (time.perf_counter() - cls._wall_start) * 1000

        print("\n" + "═" * 70)
        print("  📊  KQNODES PROFILER — REPORTE DE RENDIMIENTO")
        print("═" * 70)
        print(f"  Tiempo total de pared : {wall:>10.1f} ms")
        if cls._k_values:
            print(f"  Valores de k usados   : {cls._k_values}")
        print()

        # ── Tabla de tiempos ──────────────────────────────────────────────────
        print(f"  {'Método':<35} {'Calls':>6} {'Total ms':>10} {'Avg ms':>9} {'P95 ms':>9} {'Max ms':>9}")
        print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*9} {'-'*9} {'-'*9}")

        sorted_timers = sorted(
            cls._timers.items(),
            key=lambda x: x[1].total_ms,
            reverse=True
        )

        for name, t in sorted_timers:
            pct = (t.total_ms / wall * 100) if wall > 0 else 0
            bar = "█" * int(pct / 5)
            print(
                f"  {name:<35} {t.calls:>6} {t.total_ms:>10.1f} "
                f"{t.avg_ms:>9.2f} {t.p95_ms:>9.2f} {t.max_ms:>9.2f}"
                f"  {bar} {pct:.0f}%"
            )

        print()

        # ── Cache ─────────────────────────────────────────────────────────────
        for cache_name in set(list(cls._cache_hits.keys()) + list(cls._cache_misses.keys())):
            hits = cls._cache_hits[cache_name]
            misses = cls._cache_misses[cache_name]
            total = hits + misses
            ratio = hits / total * 100 if total else 0
            print(f"  Cache '{cache_name}': {hits} hits / {misses} misses → {ratio:.1f}% hit rate")

        print()

        # ── Candidatos ────────────────────────────────────────────────────────
        if cls._candidates_counts:
            arr = np.array(cls._candidates_counts)
            print(f"  Candidatos generados  : avg={arr.mean():.1f}  min={arr.min()}  max={arr.max()}")

        print()

        # ── Diagnóstico automático ────────────────────────────────────────────
        print("  🔍 DIAGNÓSTICO AUTOMÁTICO")
        print("  " + "-" * 50)

        if sorted_timers:
            top_name, top_t = sorted_timers[0]
            top_pct = top_t.total_ms / wall * 100 if wall > 0 else 0
            print(f"  ⚠️  Cuello de botella principal : '{top_name}' ({top_pct:.0f}% del tiempo)")

            recommendations = {
                "algorithm": (
                    "El algoritmo Q base domina. Considera:\n"
                    "    → Memoización de funcion_submodular por firma de vértices\n"
                    "    → Poda temprana si emd_iteracion == 0\n"
                    "    → Paralelizar evaluación de deltas con concurrent.futures"
                ),
                "_evaluate_k_partition": (
                    "La evaluación EMD domina. Considera:\n"
                    "    → Cache de distribuciones marginales por firma de bloque\n"
                    "    → Reducir candidatos antes de evaluar (filtro más agresivo)\n"
                    "    → Aproximar EMD con distancia L1 como pre-filtro"
                ),
                "_generate_k_cuts": (
                    "La generación de cortes domina. Considera:\n"
                    "    → Limitar combinaciones con C(internal, k-1) poda\n"
                    "    → Beam search: mantener solo top-B cortes por nivel\n"
                    "    → Cachear árbol entre llamadas con mismo alcance/mecanismo"
                ),
                "funcion_submodular": (
                    "La función submodular domina. Considera:\n"
                    "    → Incrementar uso de memoria_delta\n"
                    "    → Vectorizar evaluación de deltas candidatos\n"
                    "    → Early exit si diferencia supera umbral de la mejor partición"
                ),
            }

            for key, msg in recommendations.items():
                if key in top_name:
                    print(f"  💡 {msg}")
                    break
            else:
                print(f"  💡 Revisar manualmente '{top_name}'")

        # ── Resumen de complejidad ────────────────────────────────────────────
        print()
        print("  📐 COMPLEJIDAD ESPERADA CON 25 NODOS (50 vértices)")
        print("  " + "-" * 50)
        print("  algorithm()         : O(n³) ≈ 125,000 ops por ejecución")
        print("  _generate_k_cuts()  : O(C(2n-2, k-1)) — crece rápido con k")
        print(f"  C(48, 2) = 1,128   C(48, 3) = 17,296   C(48, 4) = 194,580")
        print("  ⚠️  Para k=5 con n=25: considera limitar internal nodes o usar beam search")
        print()
        print("═" * 70)

    @classmethod
    def reset(cls) -> None:
        cls._timers.clear()
        cls._cache_hits.clear()
        cls._cache_misses.clear()
        cls._candidates_counts.clear()
        cls._k_values.clear()
        cls._wall_start = time.perf_counter()