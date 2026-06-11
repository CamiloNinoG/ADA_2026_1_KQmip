from typing import List, Union

class TreeNode:

    def __init__(self, members):
        self.members = list(members)
        self.left = None
        self.right = None
        self.parent = None
        
    def get_all_leaves(self):
        if self.left is None and self.right is None:
            return self.members.copy()

        leaves = []

        if self.left:
            leaves.extend(self.left.get_all_leaves())

        if self.right:
            leaves.extend(self.right.get_all_leaves())

        return leaves
    
    def _all_tree_members(self):
        return self.tree.get_all_leaves()

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def get_all_leaves(self):

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