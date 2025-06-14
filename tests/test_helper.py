from falcomchain.helper import save_tree_class, load_tree_class
from falcomchain.tree import SpanningTree
import pickle
import networkx as nx

import pytest
from pathlib import Path


def test_save_tree_class(spanningtree_with_forced_root):
    
    tree = spanningtree_with_forced_root.graph 

    # save it using the function that will be tested
    save_tree_class(tree)
    
    # load it back
    path_1 = '/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/falcomchain/prepared_data/tree/tree.pkl'
    path_2 = '/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/falcomchain/prepared_data/tree/attributes.pkl'    
    with open(path_1, 'rb') as file:
        loaded_tree = pickle.load(file)
    with open(path_2, 'rb') as file:
        attr = pickle.load(file)

    # check whether everything is saved safely
    assert attr['root'] == tree.root
    assert attr['ideal_pop'] == tree.ideal_pop
    assert attr['n_teams'] == tree.n_teams
    assert attr['epsilon'] == tree.epsilon
    assert attr['supertree'] == tree.supertree
    assert attr['two_sided'] == tree.two_sided
    assert attr['tot_candidates'] == tree.total_cand
    assert attr['capacity_level'] == tree.capacity_level
    
    assert  nx.is_isomorphic(tree.graph, loaded_tree, node_match=nx.algorithms.isomorphism.categorical_node_match([], []),
                                edge_match=nx.algorithms.isomorphism.categorical_edge_match([], [])) == True
    
    
    
def test_load_tree_class(spanningtree_with_forced_root):
    
    tree = spanningtree_with_forced_root
    
    # save it using the function that is already tested.
    save_tree_class(tree)

    # load it using the function that will be tested.
    loaded_tree = load_tree_class()
    
    # check whether everything is loaded safely
    assert loaded_tree.root == tree.root
    assert loaded_tree.ideal_pop == tree.ideal_pop
    assert loaded_tree.n_teams == tree.n_teams
    assert loaded_tree.n_teams == tree.epsilon
    assert loaded_tree.supertree == tree.supertree
    assert loaded_tree.two_sided == tree.two_sided
    assert loaded_tree.tot == tree.tot_candidates
    assert loaded_tree.capacity_level == tree.capacity_level