# tree_search.py

from typing import List, Tuple
from src.strategies.kqnodes.tree_node import TreeNode

def build_tree_from_fusions(fusion_history: List) -> TreeNode | None:
    """
    Reconstruye la estructura del árbol binario de fusiones (dendrograma) a partir del historial de fusiones 
    generado secuencialmente durante la ejecución del algoritmo Q submodular.

    El proceso opera mediante las siguientes etapas:
    
    1. Indexación y Mapeo:
       - Se itera secuencialmente sobre el historial, extrayendo los grupos fusionados izquierdo, derecho y su unión.
       - Se asocian claves inmutables (tuplas ordenadas) a cada nodo para garantizar la unicidad de las referencias.

    2. Construcción de Relaciones de Parentesco:
       - Si un subgrupo izquierdo o derecho no existe en el mapa de nodos, se instancia como un nuevo TreeNode de hojas.
       - Se crea el nodo padre para representar la fusión, enlazando sus punteros izquierdo y derecho y 
         actualizando las referencias de parentesco hacia arriba.

    Args:
        fusion_history (List): Historial secuencial de fusiones (tuplas con grupo izquierdo, derecho y fusión).

    Returns:
        TreeNode | None: El nodo raíz que encabeza el árbol binario de fusiones jerárquico reconstruido.
    """
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
    """
    Colecta recursivamente todos los nodos internos del árbol que representen agrupamientos verdaderos 
    (es decir, nodos que tengan punteros izquierdo y derecho activos, descartando hojas individuales).

    Args:
        node (TreeNode): Nodo inicial desde el cual iniciar el recorrido recursivo.
        res (List[TreeNode]): Lista acumuladora donde se agregarán las referencias de los nodos internos encontrados.
    """
    if node is None:
        return
    if node.left and node.right:
        res.append(node)
    collect_internal_nodes(node.left, res)
    collect_internal_nodes(node.right, res)

def is_ancestor(ancestor: TreeNode, node: TreeNode) -> bool:
    """
    Verifica si un nodo TreeNode específico actúa como ancestro directo o indirecto 
    de otro nodo en la jerarquía del árbol de fusiones.

    El proceso sube de manera iterativa por el puntero 'parent' del nodo objetivo 
    hasta alcanzar la raíz o encontrar el ancestro buscado.

    Args:
        ancestor (TreeNode): Referencia del nodo que se sospecha es ancestro.
        node (TreeNode): Referencia del nodo descendiente a validar.

    Returns:
        bool: True si ancestor es ancestro de node, False de lo contrario o si es nulo.
    """
    current = node.parent
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False

def find_all_frontiers(node: TreeNode, k: int) -> List[List[List[Tuple[int, int]]]]:
    """
    Encuentra todas las particiones jerárquicas exactas de las hojas del árbol en exactamente k bloques
    donde cada bloque está representado por las hojas de un subárbol completo (corte de frontera).
    Esta aproximación recursiva de programación dinámica garantiza cubrir todos los cortes consistentes 
    con la estructura del árbol de fusiones del algoritmo Q de forma exacta.

    El proceso recursivo se basa en el principio de división de cuota:

    1. Caso Base (corte trivial):
       - Si k es 1, la única frontera posible es el subárbol completo en su totalidad, retornando las hojas de este nodo.
       - Si el nodo es una hoja pero k es mayor a 1, no es posible seguir dividiendo, retornando una lista vacía.

    2. División de Cuota de Bloques:
       - Se divide recursivamente la cuota k entre los hijos izquierdo y derecho (k_izq + k_der = k), probando todos los
         repartos de enteros positivos posibles.
       - Se combinan de forma cartesiana las fronteras parciales válidas obtenidas en cada subárbol para construir las 
         k-particiones finales del nodo actual.

    Args:
        node (TreeNode): Nodo actual sobre el cual evaluar la frontera recursiva.
        k (int): Cuota exacta de bloques disjuntos en los que se debe particionar el árbol.

    Returns:
        List[List[List[Tuple[int, int]]]]: Lista de particiones válidas. Cada partición contiene k bloques,
        y cada bloque contiene una lista de hojas (tuplas de tiempo e índice).
    """
    if k == 1:
        return [[node.get_all_leaves()]]
    if node is None or node.is_leaf:
        return []

    partitions = []
    for k_left in range(1, k):
        k_right = k - k_left
        left_parts = find_all_frontiers(node.left, k_left)
        right_parts = find_all_frontiers(node.right, k_right)
        for lp in left_parts:
            for rp in right_parts:
                partitions.append(lp + rp)
    return partitions
