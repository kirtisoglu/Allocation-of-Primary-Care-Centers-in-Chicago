



from collections import deque








class Part:
    """
    Partition represents a partition of the nodes of the graph. It will perform
    the first layer of computations at each step in the Markov chain - basic
    aggregations and calculations that we want to optimize.

    :ivar graph: The underlying graph.
    :type graph: :class:`~gerrychain.Graph`
    :ivar assignment: Maps node IDs to district IDs.
    :type assignment: :class:`~gerrychain.assignment.Assignment`
    :ivar parts: Maps district IDs to the set of nodes in that district.
    :type parts: Dict
    :ivar subgraphs: Maps district IDs to the induced subgraph of that district.
    :type subgraphs: Dict
    """ 

    __slots__ = (
        "id",
        "candidates",
        "center",
        "radius",
    )


    def __init__(
        self,
        cand_col: str,
        id,
        subgraph,
        travel_times,
        candidates=None,
        center=None,
        radius=None,
    ):
        """
        :subgraph:
        :param candidates: 
        :param center: 
        :param radius: 
        :param travel_times:
        :param cand_col:
        """
        
        self.subgraph = subgraph
        self.cand_col = cand_col
        self.id = id
        self.candidates = self._part_candidates(subgraph)
        self.center, self.radius = self._part_center(travel_times)
        
        

    def _part_candidates(self):
        return [node for node in self.subgraph.nodes if self.subgraph.nodes[node][self.cand_col]==True]



    def _part_center(self, travel_times):
        
        radius = 100000
        queue = deque()
        
        while queue:
            candidate = queue.pop()
            candidate_max = max(travel_times[(node, candidate)] for node in self.subgraph.nodes)
            if candidate_max < radius:
                center = candidate
                radius = candidate_max

        return center, radius
    





"""
    def __getitem__(self, key: str) -> Any:
        
        Allows accessing the values of updaters computed for this
        Partition instance.

        :param key: Property to access.
        :type key: str

        :returns: The value of the updater.
        :rtype: Any
        
        if key not in self._cache:
            self._cache[key] = self.updaters[key](self)
        return self._cache[key]

    def __getattr__(self, key):
        return self[key]

    def keys(self):
        return self.updaters.keys()
    
    
    def __repr__(self):
        number_of_parts = len(self)
        s = "s" if number_of_parts > 1 else ""
        return "<{} [{} part{}]>".format(self.__class__.__name__, number_of_parts, s)

    
    def __len__(self):
        return len(self.parts)

    
    def __iter__(self):
        return self.keys()"""







