//! Cascading challenge tree for parallel proof generation
//!
//! Dory-Prime's key innovation is precomputing all 2^sigma possible challenge
//! paths in a binary tree structure. This allows the prover to compute all
//! round messages in parallel, rather than sequentially.

use crate::primitives::arithmetic::{Field, Group, PairingCurve};
use std::marker::PhantomData;

/// A node in the cascading tree at a specific depth
#[derive(Clone)]
pub struct TreeNode<E: PairingCurve> {
    /// The accumulated challenge product for this path
    pub challenge_product: <E::G1 as Group>::Scalar,
    /// Index of this node at its depth (0 to 2^depth - 1)
    pub index: usize,
}

impl<E: PairingCurve> TreeNode<E> {
    /// Create a new tree node
    pub fn new(challenge_product: <E::G1 as Group>::Scalar, index: usize) -> Self {
        Self {
            challenge_product,
            index,
        }
    }

    /// Create the root node (depth 0, no challenges yet)
    pub fn root() -> Self {
        Self {
            challenge_product: <E::G1 as Group>::Scalar::one(),
            index: 0,
        }
    }
}

/// Messages stored at each tree node
#[derive(Clone)]
pub struct NodeMessages<E: PairingCurve> {
    /// Beta challenge used
    pub beta: <E::G1 as Group>::Scalar,
    /// Alpha challenge used
    pub alpha: <E::G1 as Group>::Scalar,
}

/// One level of the tree
#[derive(Clone)]
pub struct TreeLevel<E: PairingCurve> {
    /// Nodes at this level
    pub nodes: Vec<TreeNode<E>>,
    /// Messages at each node
    pub messages: Vec<NodeMessages<E>>,
}

/// Cascading challenge tree for Dory-Prime
///
/// This tree precomputes all possible challenge paths, enabling fully
/// parallel proof generation.
#[derive(Clone)]
pub struct CascadingTree<E: PairingCurve> {
    /// Depth of the tree (sigma = number of variables)
    depth: usize,
    /// Number of rounds (max(nu, sigma))
    num_rounds: usize,
    /// All levels of the tree
    levels: Vec<TreeLevel<E>>,
    /// Phantom for the curve
    _phantom: PhantomData<E>,
}

impl<E: PairingCurve> CascadingTree<E>
where
    <E::G1 as Group>::Scalar: Field,
{
    /// Create a new empty cascading tree
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            num_rounds: depth,
            levels: vec![TreeLevel {
                nodes: vec![TreeNode::root()],
                messages: vec![],
            }],
            _phantom: PhantomData,
        }
    }

    /// Build the tree structure for a given number of rounds
    ///
    /// This creates the full binary tree structure without computing
    /// actual messages. The actual challenge computation requires
    /// the forkable prover state and transcript.
    pub fn build_for_rounds(num_rounds: usize) -> Self {
        let depth = num_rounds;
        let mut levels = vec![TreeLevel {
            nodes: vec![TreeNode::root()],
            messages: vec![],
        }];

        for level in 0..num_rounds {
            let num_parent_nodes = levels[level].nodes.len();
            let mut level_nodes = Vec::with_capacity(num_parent_nodes * 2);

            for i in 0..num_parent_nodes {
                // Child 0: challenge 0 path
                level_nodes.push(TreeNode::new(<E::G1 as Group>::Scalar::one(), i * 2));
                // Child 1: challenge 1 path
                level_nodes.push(TreeNode::new(<E::G1 as Group>::Scalar::one(), i * 2 + 1));
            }

            levels.push(TreeLevel {
                nodes: level_nodes,
                messages: vec![NodeMessages {
                    beta: <E::G1 as Group>::Scalar::one(),
                    alpha: <E::G1 as Group>::Scalar::one(),
                }; levels[0].nodes.len() * 2],
            });
        }

        Self {
            depth,
            num_rounds,
            levels,
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
        if level >= self.levels.len() {
            0
        } else {
            self.levels[level].nodes.len()
        }
    }

    /// Get the challenge product for a specific node
    pub fn get_challenge_product(&self, level: usize, index: usize) -> Option<<E::G1 as Group>::Scalar> {
        self.levels
            .get(level)
            .and_then(|l| l.nodes.get(index))
            .map(|n| n.challenge_product)
    }

    /// Get all leaf nodes (depth = num_rounds)
    pub fn get_leaf_nodes(&self) -> &[TreeNode<E>] {
        if self.levels.is_empty() {
            &[]
        } else {
            &self.levels[self.levels.len() - 1].nodes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::arkworks::BN254;

    #[test]
    fn test_tree_depth() {
        let tree = CascadingTree::<BN254>::new(3);
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
        assert_eq!(tree.nodes_at_level(0), 1);
        assert_eq!(tree.nodes_at_level(1), 2);
        assert_eq!(tree.nodes_at_level(2), 4);
        assert_eq!(tree.nodes_at_level(3), 8);
    }
}
