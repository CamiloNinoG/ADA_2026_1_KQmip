# combinatorics.py

def stirling_segundo_tipo(n: int, k: int) -> int:
    """
    Calcula el número de Stirling del segundo tipo S(n, k) mediante una aproximación recursiva optimizada con memoización dinámica. 
    Este valor representa el tamaño combinatorio exacto para dividir un conjunto de n elementos en k bloques disjuntos no vacíos, 
    permitiendo estimar el tamaño del espacio de búsqueda para determinar la viabilidad de una búsqueda exhaustiva.

    El proceso se desarrolla bajo las siguientes consideraciones matemáticas:

    1. Caso Base y Frontera:
       - Si k es igual a 1, solo existe una única forma de partición (colocar todos los elementos en el mismo bloque).
       - Si n es igual a k, cada elemento debe ir en su propio bloque individual, existiendo solo 1 combinación.
       - Si n es menor que k, o si k es menor que 1, la partición es físicamente imposible, retornando 0.

    2. Paso Recursivo con Memoización:
       - Se calcula la relación recurrente: S(n, k) = k * S(n - 1, k) + S(n - 1, k - 1).
       - Las sub-soluciones se guardan en un diccionario de caché indexado por la tupla (n_val, k_val) para evitar el recalculo exponencial de subproblemas repetidos.

    Args:
        n (int): Número total de elementos del conjunto (las variables futuras o presentes del sistema).
        k (int): Número de bloques disjuntos en los cuales se desea particionar el conjunto.

    Returns:
        int: El número total de k-particiones posibles para un conjunto de n variables.
    """
    memo = {}
    
    def solve(n_val, k_val):
        if k_val == 1 or n_val == k_val:
            return 1
        if n_val < k_val or k_val < 1:
            return 0
        state = (n_val, k_val)
        if state not in memo:
            memo[state] = k_val * solve(n_val - 1, k_val) + solve(n_val - 1, k_val - 1)
        return memo[state]
        
    return solve(n, k)
