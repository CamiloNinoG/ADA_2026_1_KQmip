# tree_search.py

from typing import List, Tuple
from src.strategies.kqnodes.tree_node import TreeNode

def build_tree_from_fusions(fusion_history: List) -> TreeNode | None:
    """Reconstruye el árbol binario de fusiones a partir del historial de fusiones."""
    node_map = {}
    root = None
    for left, right, fusion in fusion_history:
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

def collect_internal_nodes(node: TreeNode, res: List[TreeNode]):
    """Colecta recursivamente todos los nodos internos (que tienen hijos izquierdo y derecho)."""
    if node is None:
        return
    if node.left and node.right:
        res.append(node)
    collect_internal_nodes(node.left, res)
    collect_internal_nodes(node.right, res)

def is_ancestor(ancestor: TreeNode, node: TreeNode) -> bool:
    """Verifica si el nodo ancestor es ancestro de node."""
    current = node.parent
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False

def find_all_frontiers(node: TreeNode, k: int) -> List[List[List[Tuple[int, int]]]]:
    """
    Encuentra todas las particiones válidas de las hojas del árbol en exactamente k bloques
    donde cada bloque corresponde a las hojas de un subárbol completo (frontera del árbol).
    Retorna una lista de particiones, donde cada partición es una lista de bloques,
    y cada bloque es una lista de hojas (tuplas (tiempo, idx)).
    """
    if k == 1:
        return [[node.get_all_leaves()]]
    if node is None or node.is_leaf: # k > 1 no es posible en hoja
        return []

    partitions = []
    # Intenta dividir la cuota de k bloques entre los hijos izquierdo y derecho
    for k_left in range(1, k):
        k_right = k - k_left
        left_parts = find_all_frontiers(node.left, k_left)
        right_parts = find_all_frontiers(node.right, k_right)
        for lp in left_parts:
            for rp in right_parts:
                partitions.append(lp + rp)
    return partitions
