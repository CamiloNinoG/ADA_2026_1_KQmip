# tree_node.py
from typing import List, Union

class TreeNode:
    """
    Representa un nodo en el árbol binario de fusiones (dendrograma) jerárquicas
    generado por el algoritmo Q.
    """

    def __init__(self, members):
        """
        Inicializa un nuevo TreeNode.

        Args:
            members (list): Lista de variables que componen el grupo de fusión representado por este nodo.
        """
        self.members = list(members)
        self.left = None
        self.right = None
        self.parent = None
    
    def _all_tree_members(self):
        """Retorna todas las hojas pertenecientes al subárbol de este nodo."""
        return self.get_all_leaves()

    @property
    def is_leaf(self):
        """Determina si el nodo actual es una hoja (no tiene hijos)."""
        return self.left is None and self.right is None

    def get_all_leaves(self):
        """
        Recorre recursivamente el subárbol para recolectar los miembros hoja de todos sus descendientes.

        Returns:
            list: Lista de hojas de este subárbol.
        """
        if self.is_leaf:
            return list(self.members)

        leaves = []

        if self.left:
            leaves.extend(self.left.get_all_leaves())

        if self.right:
            leaves.extend(self.right.get_all_leaves())

        return leaves

    def __repr__(self):
        return f"TreeNode({len(self.get_all_leaves())})"

