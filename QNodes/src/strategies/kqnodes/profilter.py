# profilter.py
import time
import numpy as np

class KQNodesProfiler:
    """
    Perfilador y medidor de tiempos de ejecución para las funciones críticas del algoritmo KQNodes.
    Permite identificar cuellos de botella y registrar estadísticas agregadas en microsegundos y milisegundos.
    """
    _calls = {}
    _times = {}
    _k_vals = []
    _start_wall = 0.0

    @classmethod
    def wrap(cls, instance):
        """
        Envuelve de manera dinámica los métodos clave de la instancia del analizador
        para interceptar sus llamadas y registrar los tiempos de ejecución.

        Args:
            instance: Instancia de KQNodes a perfilar.
        """
        cls._start_wall = time.perf_counter()
        cls._calls.clear()
        cls._times.clear()
        cls._k_vals.clear()
        
        methods_to_wrap = [
            'aplicar_estrategia_k',
            '_find_best_partition',
            '_evaluate_k_partition',
            '_build_tree_from_fusions',
            'algorithm'
        ]
        
        for name in methods_to_wrap:
            if hasattr(instance, name):
                orig_method = getattr(instance, name)
                
                def make_wrapper(orig, m_name):
                    def wrapper(*args, **kwargs):
                        if m_name == 'aplicar_estrategia_k':
                            k_val = kwargs.get('k', args[4] if len(args) > 4 else 3)
                            cls._k_vals.append(k_val)
                            
                        t0 = time.perf_counter()
                        res = orig(*args, **kwargs)
                        t1 = time.perf_counter()
                        
                        elapsed = (t1 - t0) * 1000.0 # ms
                        
                        cls._calls[m_name] = cls._calls.get(m_name, 0) + 1
                        if m_name not in cls._times:
                            cls._times[m_name] = []
                        cls._times[m_name].append(elapsed)
                        return res
                    return wrapper
                
                setattr(instance, name, make_wrapper(orig_method, name))

    @classmethod
    def report(cls):
        """
        Imprime un reporte tabulado detallando el número de llamadas, tiempo total, promedio y
        el percentil 95 de duración de cada uno de los métodos envueltos.
        """
        total_wall = (time.perf_counter() - cls._start_wall) * 1000.0
        print("\n══════════════════════════════════════════════════════════════════════")
        print("  📊  KQNODES PROFILER — REPORTE DE RENDIMIENTO")
        print("══════════════════════════════════════════════════════════════════════")
        print(f"  Tiempo total de pared :    {total_wall:.1f} ms")
        print(f"  Valores de k usados   : {cls._k_vals}\n")
        
        print(f"  {'Método':<35} {'Calls':>6} {'Total ms':>10} {'Avg ms':>10} {'P95 ms':>10} {'Max ms':>10}")
        print("  ----------------------------------- ------ ---------- --------- --------- ---------")
        
        for name in ['aplicar_estrategia_k', 'algorithm', '_find_best_partition', '_evaluate_k_partition', '_build_tree_from_fusions']:
            if name in cls._calls:
                calls = cls._calls[name]
                times = cls._times[name]
                total = sum(times)
                avg = total / calls
                p95 = np.percentile(times, 95)
                max_val = max(times)
                
                pct = (total / total_wall) * 100.0
                bar_len = int(pct / 5)
                bar = "█" * bar_len
                
                print(f"  {name:<35} {calls:>6} {total:>10.1f} {avg:>9.2f} {p95:>9.2f} {max_val:>9.2f}  {bar} {pct:.0f}%")
        print("\n\n  🔍 DIAGNÓSTICO AUTOMÁTICO")
        print("  --------------------------------------------------")
        
        main_method = 'aplicar_estrategia_k'
        if main_method in cls._times:
            total_main = sum(cls._times[main_method])
            pct_main = (total_main / total_wall) * 100.0
            print(f"  ⚠️  Cuello de botella principal : '{main_method}' ({pct_main:.0f}% del tiempo)")
            print(f"  💡 Revisar manualmente '{main_method}'\n")
