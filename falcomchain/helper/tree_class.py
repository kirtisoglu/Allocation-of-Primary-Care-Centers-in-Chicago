import pickle
from typing import Optional
from falcomchain.tree import SpanningTree


def save_tree_class(tree: SpanningTree, path_1: Optional[str]=None, path_2: Optional[str]=None):
    
    if path_1 == None:
        path_1 = '/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/tree.pkl'
    if path_2 == None:
        path_2 = '/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/attributes.pkl'
    
    dict = {'root': tree.root, 'ideal_pop': tree.ideal_pop, 'n_teams': tree.n_teams, 'epsilon':tree.epsilon, 'column_names': tree.column_names,
            'supergraph':tree.supertree, 'two_sided': tree.two_sided, 'tot_candidates': tree.total_cand, 'capacity_level': tree.capacity_level}
    
    with open(path_1, 'wb') as file:
        pickle.dump(tree.graph, file)
        
    with open(path_2, 'wb') as file:
        pickle.dump(dict, file)
  
        
def load_tree_class(path_1: Optional[str]=None, path_2: Optional[str]=None): 
    
    if path_1 == None:
        path_1 = '/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/tree.pkl'
    if path_2 == None:
        path_2 = '/Users/kirtisoglu/Documents/Documents/GitHub/Allocation-of-Primary-Care-Centers-in-Chicago/data/processed/attributes.pkl'   
    
    with open(path_1, 'rb') as file:
        loaded_tree = pickle.load(file)
    
    with open(path_2, 'rb') as file:
        attr = pickle.load(file)
        
    return SpanningTree(graph=loaded_tree, ideal_pop=attr['ideal_pop'], epsilon=attr['epsilon'], n_teams=attr['n_teams'], 
                        capacity_level=attr['capacity_level'], column_names=attr['column_names'], two_sided=attr['two_sided'], 
                        supergraph=attr['supergraph'])


    
