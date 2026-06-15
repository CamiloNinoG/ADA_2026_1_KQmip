# combinatorics.py

def stirling_segundo_tipo(n: int, k: int) -> int:
    """
    Calcula el número de Stirling del segundo tipo S(n, k) usando memoización dinámica.
    Representa el número de formas de particionar un conjunto de n elementos en k bloques no vacíos.
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
