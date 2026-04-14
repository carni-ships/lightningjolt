//! Cascading challenge tree for parallel proof generation
//!
//! Dory-Prime's key innovation is precomputing all 2^sigma possible challenge
//! paths in a binary tree structure. This allows the prover to compute all
//! round messages in parallel, rather than sequentially.
//!
//! ## Tree Structure
//!
//! At depth 0, we have 1 root node (the starting point).
//! At depth 1, we have 2 nodes (for challenge 0 or 1 at first position).
//! At depth 2, we have 4 nodes (for challenges 00, 01, 10, 11), etc.
//! At depth sigma, we have 2^sigma leaf nodes.
//!
//! Each node contains the accumulated challenge product for that path.

use crate::primitives::arithmetic::{Field, PairingCurve};
use std::marker::PhantomData;

/// The scalar field type for a pairing curve
pub type Scalar<E> = <<E as PairingCurve>::G1 as crate::primitives::arithmetic::Group>::Scalar;

/// A node in the cascading tree at a specific depth
#[derive(Clone)]
pub struct TreeNode<E: PairingCurve> {
    /// The accumulated challenge product for this path
    pub challenge_product: Scalar<E>,
    /// Index of this node at its depth (0 to 2^depth - 1)
    pub index: usize,
}

impl<E: PairingCurve> TreeNode<E> {
    /// Create a new tree node
    pub fn new(challenge_product: Scalar<E>, index: usize) -> Self {
        Self {
            challenge_product,
            index,
        }
    }

    /// Create the root node (depth 0, no challenges yet)
    pub fn root() -> Self {
        Self {
            challenge_product: Scalar::<E>::one(),
            index: 0,
        }
    }
}

/// Cascading challenge tree for Dory-Prime
///
/// This tree precomputes all possible challenge paths, enabling fully
/// parallel proof generation. At each depth, we store messages for both
/// possible challenge values (0 and 1).
#[derive(Clone)]
pub struct CascadingTree<E: PairingCurve> {
    /// Depth of the tree (sigma = number of variables)
    depth: usize,
    /// Number of rounds (max(nu, sigma))
    num_rounds: usize,
    /// Nodes at each level: Vec<level> -> Vec<Node>
    nodes: Vec<Vec<TreeNode<E>>>,
    /// Field and curve marker
    _phantom: PhantomData<E>,
}

impl<E: PairingCurve> CascadingTree<E> {
    /// Create a new empty cascading tree
    pub fn new_empty(depth: usize) -> Self {
        Self {
            depth,
            num_rounds: depth,
            nodes: vec![vec![TreeNode::root()]],
            _phantom: PhantomData,
        }
    }

    /// Returns the depth of the tree (sigma)
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the number of rounds
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Returns the number of nodes at a given level
    pub fn nodes_at_level(&self, level: usize) -> usize {
        if level >= self.nodes.len() {
            0
        } else {
            self.nodes[level].len()
        }
    }

    /// Get the challenge product for a specific path
    pub fn get_challenge_product(&self, level: usize, index: usize) -> Option<Scalar<E>> {
        self.nodes
            .get(level)
            .and_then(|l| l.get(index))
            .map(|n| n.challenge_product)
    }

    /// Get all leaf nodes (depth = num_rounds)
    pub fn get_leaf_nodes(&self) -> &[TreeNode<E>] {
        if self.nodes.is_empty() {
            &[]
        } else {
            let leaf_depth = (self.depth).min(self.nodes.len() - 1);
            &self.nodes[leaf_depth]
        }
    }

    /// Get the nodes at a specific level
    pub fn get_level(&self, level: usize) -> Option<&[TreeNode<E>]> {
        self.nodes.get(level).map(|l| l.as_slice())
    }

    /// Build the tree structure for a given number of rounds
    ///
    /// This creates a placeholder tree. The full implementation would
    /// precompute all possible challenge paths.
    pub fn build_for_rounds(num_rounds: usize) -> Self {
        let depth = num_rounds;
        let mut nodes = vec![vec![TreeNode::<E>::root()]];

        // Build the full binary tree structure
        for level in 0..num_rounds {
            let num_parent_nodes = nodes[level].len();
            let mut level_nodes = Vec::with_capacity(num_parent_nodes * 2);

            for parent in &nodes[level] {
                // Child 0: takes challenge 0
                level_nodes.push(TreeNode::new(parent.challenge_product, level_nodes.len()));
                // Child 1: takes challenge 1
                level_nodes.push(TreeNode::new(parent.challenge_product, level_nodes.len()));
            }

            nodes.push(level_nodes);
        }

        Self {
            depth,
            num_rounds,
            nodes,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::arkworks::BN254;

    #[test]
    fn test_tree_depth() {
        let tree = CascadingTree::<BN254>::new_empty(3);
        assert_eq!(tree.depth(), 3);
    }

    #[test]
    fn test_root_node() {
        let root = TreeNode::<BN254>::root();
        assert_eq!(root.index, 0);
    }

    #[test]
    fn test_tree_structure() {
        let tree = CascadingTree::<BN254>::build_for_rounds(3);
        // Level 0: 1 node
        assert_eq!(tree.nodes_at_level(0), 1);
        // Level 1: 2 nodes
        assert_eq!(tree.nodes_at_level(1), 2);
        // Level 2: 4 nodes
        assert_eq!(tree.nodes_at_level(2), 4);
        // Level 3: 8 nodes (leaf)
        assert_eq!(tree.nodes_at_level(3), 8);
    }
}
