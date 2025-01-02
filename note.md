
Reamining files to scan  ->  partition,  constraints,  graph,  meta,  helper,  prepared_data,  travel_time,  vendor,  initial files


-----------------  PROCESSES   -----------------

1. Preparation: Datadan graph olustur...
2. Initial Solution: 
3. Iteration:
4. Stats:
5. Optimization


-----------------  CLASSES   -----------------

Validity
Updater
Constraints
Metagraph
Graph
DataHandler
Plot
Stats

MarkovChain

Partition
Assignment

RecursiveCut
SpanningTree



-----------------  MAIN STREAM   -----------------

optimization ->

chain.py  -> Markov chain is defined here. For each state, Recom class is called from proposals.tree_proposals module.

proposals -> Recomb class in tree_proposals file is the method to be used for the intermadiate steps of the chain. 
             It calls capacitated_recursive tree from tree.py and Partition class form partition file in the partition folder.

partition -> There are 4 files here but it is enough to mention two of them for the main stream.
                
                partition:
                assignment:

tree.py  -> We need to understand three functions here:

                capacitated_hierarchical_tree:
                bipartition_tree:
                find_edge_cuts:


updaters -> We use only two files here:

                cut_edges:
                flows:

Order: SingleMetricOptimizers by user -> MarkovChain by tilted_run function in SingleMetricOptimizers 
        -> recomb function at every state by MarkovChain. Remark: parameters of recomb is defined by user, not in MarkovChain. -> capacitated_recursive_tree by recomb
        -> bipartition_tree -> find_edge_cuts -> find_edge_cuts ->

-----------------  OTHERS   -----------------

graph

metrics

meta

vendor

constraints

metagraph.py




---------------------------------------------------------------- GRAPH ----------------------------------------------------------------





Real Graph

    1. Density of a node attribute olsun.



Transportatiom network - input: street data,census blocks, hospitals, otherlocations?, GTFS?

    0.  Advance plotting functions!

    1. Clean openstreetmap data.
    
        Read OSM data with custom filter. Get rid of unnecessary columns. 

    2. Create the graph.
    
    2. Filtering data based on bounding box

        We need a bounding box to extract a subdata extract only a subset of the whole OSM PBF file covering e.g. a specific region. 
        The bounding box should contain all transportation network in the city and be a bit larger than the city area for a precise
        travel time calculation. Because there might be a shorter travel route in time exceeding the city boundaries since transportation
        network most likely exceeds the city boundaries.
        
        We handle this using Pyrosm library, which provides an easy way to filter larger PBF files using a rectangular shape or a more 
        complex geometric feature, e.g. a Polygon. Pyrosm requries a list of coordinates

    https://networkx.org/documentation/stable/auto_examples/geospatial/plot_osmnx.html#sphx-glr-auto-examples-geospatial-plot-osmnx-py


---------------------------------------------------------------- TRAVEL TIME ----------------------------------------------------------------

1. 



---------------------------------------------------------------- LOCAL SEARCH ----------------------------------------------------------------

1. There are two ipynb files for that.



---------------------------------------------------------------- MARKOV CHAIN ----------------------------------------------------------------





---------------------------------------------------------------- TESTING REPO ----------------------------------------------------------------



---------------------------------------------------------------- CODE OPTIMIZATION ------------------------------------------------------------

# 1. Profile Your Code

Code profiling is the process of analyzing your program's behavior during runtime to identify performance bottlenecks and resource usage patterns and improve the performance of your software.It provides detailed insights into:
* Execution time of functions/methods
* Memory allocation and usage
* CPU utilization
* Function call frequencies and durations
Choose a profiling tool. Python: cProfile, line$\_$profiler. Execute your program under the profiler's control. Analyze 
* Functions that consume the most time
* Frequently called functions
* Memory-intensive operations
* CPU-intensive sections
Identify bottlenecks looking for
* Functions with unexpectedly long execution times
* Excessive memory allocations
* Inefficient algorithms or data structures
Focus on improving the performance of the identified bottlenecks. Once bottlenecks are identified:
1. Refactor inefficient algorithms
2. Optimize database queries
3. Implement caching where appropriate
4. Reduce unnecessary function calls or computations
5. Optimize memory usage and management
Profiling should be an ongoing process, especially as your codebase evolves and grows. Regular profiling helps maintain optimal performance over time.

To profile a function that takes a single argument and you can do:

```bash
import cProfile
import re
cProfile.run('re.compile("foo|bar")')
```

to run re.compile() and print profile results. See a typical result table here: https://docs.python.org/3/library/profile.html
The first line indicates that 214 calls were monitored. Of those calls, 207 were primitive, meaning that the call was not induced 
via recursion. The next line Ordered by: cumulative time indicates the output is sorted by the cumtime values. The column headings include:

ncalls
for the number of calls.

tottime
for the total time spent in the given function (and excluding time made in calls to sub-functions)

percall
is the quotient of tottime divided by ncalls

cumtime
is the cumulative time spent in this and all subfunctions (from invocation till exit). This figure is accurate even for recursive functions.

percall
is the quotient of cumtime divided by primitive calls

filename:lineno(function)
provides the respective data of each function




# 2. Use Cache nad __slots__ for Memory Usage

 Memory usage and runtime are often interrelated. While focusing on runtime, you should be mindful of memory usage because:

    - Excessive memory usage can lead to swapping, which severely impacts runtime.
    - Efficient memory use often correlates with better cache utilization, improving runtime.
    - Some optimization techniques (like caching) trade memory for speed.

A caching example over graphs:

This caches the results of subgraph computations, speeding up repeated accesses. 
it's generally beneficial to put all anticipated attributes in `__slots__`. This approach:

- Reduces memory usage
- Slightly improves attribute access speed
- Prevents accidental creation of new attributes

However, be aware that:

- You can't add new attributes dynamically
- Multiple inheritance becomes more complex
- Some libraries might not work well with `__slots__`

from functools import lru_cache

```bash
class Graph:
    def __init__(self):
        self.nodes = {}

    @lru_cache(maxsize=1000)
    def get_subgraph(self, node_id, depth):
        # Expensive computation to retrieve subgraph
        pass
```

- If the subgraph depends on the specific Graph instance, you can modify the caching to include the instance:

```bash
class Graph:
    def __init__(self):
        self.nodes = {}

    def get_subgraph(self, node_id, depth):
        return self._cached_get_subgraph(self, node_id, depth)

    @staticmethod
    @lru_cache(maxsize=1000)
    def _cached_get_subgraph(graph_instance, node_id, depth):
        # Expensive computation to retrieve subgraph
        pass
```

Yield is more efficient, memory-wise, and also sometimes execution-wise. If you iterate over a list 
of 1,000,000 elements, Python has to generate the entire list and store the contents in memory before 
beginning the first iteration. With a generator (using yield), the elements are created at the time of 
iteration, so 1,000,000 elements don’t need to be pre-calculated first and stored in memory.


# 3. Optimize Tree Traversals 

This reduces redundant traversals and can significantly improve efficiency, especially for large graphs
when we calculate traversals many times in an iteration.

For example, 

Successors function in NetworkX library uses breath-first search. If you are calling this function and
coding an algorithm similar to breath-first search to calculate some values for a tree, then do not call 
the function from the library. Write the function yourself explicitly and handle the value calculations within
that code block.

# 4. Parallelize Your Markov Chain

- Run multiple chains in parallel and aggregate results.
- Use parallel tempering, running chains at different temperatures.
- Parallelize expensive computations within each step (e.g., constraint checks).
- Use libraries like multiprocessing or concurrent.futures for parallelization.

from multiprocessing import Pool
 
def run_chain(seed):
    chain = MarkovChain(seed)
    return chain.run()

if __name__ == '__main__':
    with Pool(processes=4) as pool:
       results = pool.map(run_chain, range(4)) 

Remember to profile your code to identify the most impactful optimizations for your specific use case.


# 5. Absorption Time Optimization

- Track the frequency of transitions leading to desired states.
- Implement an adaptive proposal distribution that favors transitions to these states.
- Use simulated annealing: start with a high "temperature" allowing more random moves, then gradually "cool" to focus on more promising states.

```bash
import random
import math

class AdaptiveMarkovChain:
    def __init__(self, initial_state):
        self.current_state = initial_state
        self.transition_counts = {}
        self.temperature = 1.0

    def step(self):
        proposed_state = self.propose_next_state()
        if self.accept_proposal(proposed_state):
            self.current_state = proposed_state
        self.update_transition_counts(self.current_state)
        self.cool_temperature()

    def propose_next_state(self):
        # Use transition_counts to bias towards promising states
        pass

    def accept_proposal(self, proposed_state):
        # Use simulated annealing acceptance criterion
        return random.random() < math.exp(-self.calculate_energy_difference() / self.temperature)

    def update_transition_counts(self, state):
        self.transition_counts[state] = self.transition_counts.get(state, 0) + 1

    def cool_temperature(self):
        self.temperature *= 0.99  # Gradual cooling
```


# 6. Parallelize expensive computations within each step

Here's an example of parallelizing constraint checks in a redistricting scenario: 


```bash
from concurrent.futures import ThreadPoolExecutor

class RedistrictingChain:
    def __init__(self, graph, num_districts):
        self.graph = graph
        self.num_districts = num_districts
        self.current_partition = self.initial_partition()

    def step(self):
        proposed_partition = self.propose_new_partition()
        if self.check_constraints_parallel(proposed_partition):
            self.current_partition = proposed_partition

    def check_constraints_parallel(self, partition):
        constraints = [
            self.check_population_balance,
            self.check_contiguity,
            self.check_compactness,
            # Add more constraints as needed
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda f: f(partition), constraints))

        return all(results)

    def check_population_balance(self, partition):
        # Check if districts have roughly equal population
        pass

    def check_contiguity(self, partition):
        # Check if all districts are contiguous
        pass

    def check_compactness(self, partition):
        # Check if districts are reasonably compact
        pass
```

In this example, we parallelize the constraint checks, which can be computationally expensive. 
Each constraint check runs in a separate thread, potentially utilizing multiple CPU cores and 
reducing the overall time for constraint verification.

Remember to profile your specific use case to ensure that parallelization actually improves performance, 
as the overhead of creating and managing threads can sometimes outweigh the benefits for very quick operations.


# 7. Use cache-friendly graph representations

- Adjacency arrays instead of adjacency lists can improve cache performance for algorithms like Dijkstra's and Prim's.

- Partition the graph into cache-sized segments:
    * Divide vertex data into segments that fit in cache3.
    * Process subgraphs that fit in cache to limit random accesses3.

- Optimize data layout
- Use blocked or tiled data layouts to match access patterns of tiled algorithms4.
- Consider cache-oblivious implementations for algorithms like Floyd-Warshall

- Cache high-degree vertices:
    * In small-world networks, cache information for high-degree vertices that are frequently accessed5.
    * This can significantly reduce communication in distributed graph algorithms.

- Use cache-aware scheduling:
    * Schedule vertex processing based on what's currently in cache2.
    * Process vertices with cached neighbors together to maximize reuse.

- Employ parallel cache-aware merging:
    * When processing graph segments in parallel, use cache-aware techniques to merge results

- Profile and analyze cache performance:
    * Use tools to measure cache misses, stall cycles, and memory traffic34.
    * Optimize based on these metrics rather than just overall runtime.

Combine caching with other optimizations:
* Use caching alongside techniques like graph reordering or clustering
* Adjust caching strategies based on graph properties and algorithm characteristics.


7. Adjacency lists?




"""

import cProfile
import pstats
import io


# Create a main script that runs your Markov chain to get an overall profile of your entire chain process.


def main():
    # Your main function that sets up and runs the Markov chain
    run_markov_chain()

if __name__ == "__main__":
    # Use cProfile to run the main function
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    
    # Print sorted stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
    
    
    
    # To get more detailed information about specific functions or to filter the output, you can modify the stats printing:
    
    # Instead of print_stats(), use:
    ps.print_stats(.1)  # Print only the top 10% of functions
    # or
    ps.print_callers(.5)  # Print callers for the top 50% of functions
    # or
    ps.print_callees(.1)  # Print callees for the top 10% of functions
    
    
    # If you want to focus on specific parts of your code, you can use the Profile class directly in your code:
    
import cProfile

def some_important_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your function code here
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
    
    
# For a more interactive analysis, you can save the profiling results to a file and use tools like snakeviz:

# In your main script
    profiler.dump_stats('redistricting_profile.prof')

# Then in terminal:
# snakeviz redistricting_profile.prof



    
emember, profiling can slow down your code, so it's best to use it for a limited number of iterations 
when dealing with long-running processes like Markov chains. Modify the chain function to accept a parameter 
for the number of iterations:

def run_markov_chain(num_iterations):
    # Your Markov chain implementation here
    for i in range(num_iterations):
        # Perform one step of the Markov chain
        pass


Create a main script that uses cProfile to run the Markov chain for a limited number of iterations


import cProfile
import pstats
import io

def main(num_iterations):
    run_markov_chain(num_iterations)

if __name__ == "__main__":
    num_iterations = 1000  # Set this to your desired number of iterations

    profiler = cProfile.Profile()
    profiler.enable()
    
    main(num_iterations)
    
    profiler.disable()
    
    # Print sorted stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

If you want to profile specific parts of your Markov chain process, you can use context managers 
to enable and disable profiling for specific sections:

import cProfile
import pstats
import io

def run_markov_chain(num_iterations):
    profiler = cProfile.Profile()
    
    for i in range(num_iterations):
        # Regular non-profiled operations
        
        with profiler:
            # Profiled operations
            perform_expensive_operation()
        
        # More non-profiled operations

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())

def perform_expensive_operation():
    # Your expensive operation here
    pass

if __name__ == "__main__":
    num_iterations = 1000
    run_markov_chain(num_iterations)

If you want to profile multiple runs and get an average, you can use a function like this

def profile_average(func, num_runs, *args, **kwargs):
    total_stats = None
    for _ in range(num_runs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        func(*args, **kwargs)
        
        profiler.disable()
        
        if total_stats is None:
            total_stats = pstats.Stats(profiler)
        else:
            total_stats.add(profiler)
    
    # Calculate average
    for key in total_stats.stats:
        total_stats.stats[key] = tuple(v / num_runs for v in total_stats.stats[key])
    
    # Print average stats
    s = io.StringIO()
    total_stats.stream = s
    total_stats.sort_stats('cumulative')
    total_stats.print_stats()
    print(s.getvalue())

# Usage
num_iterations = 1000
num_runs = 5
profile_average(run_markov_chain, num_runs, num_iterations)"""

"""
By using these techniques, you'll be able to see which functions are taking 
the most time, how often they're called, and where the bottlenecks in your 
redistricting algorithm might be. This information will be invaluable for 
optimizing your code and improving the efficiency of your Markov chain process."""



"""Py-spy: This is a sampling profiler that can profile Python code without modifying it or 
impacting performance significantly. It's especially useful for profiling long-running processes 
and can handle multithreaded applications 35.
Pyinstrument: This profiler samples the call stack, making it less intrusive than cProfile. 
It provides concise reports focusing on the most time-consuming parts of your code 3.
Austin: This is a frame stack sampler for CPython that can profile running applications with 
minimal impact on performance. It has VSCode integration and supports Python 3.11 1.
Scalene: This is a high-performance CPU and memory profiler that can profile both Python and 
C code. It's particularly useful for identifying memory usage issues 2.
FunctionTrace: This tool uses the Firefox Profiler to render results to an interactive graph, 
which can be helpful for visualizing complex processes like Markov chains 3.
Palanteer: While relatively new, it can profile both Python and C++ code, which might be useful 
if your project includes C extensions 3.
For your specific case of profiling a Markov chain redistricting project:

If you need to profile long-running processes without significantly impacting performance, 
Py-spy or Austin might be good choices.
If you want detailed memory profiling alongside CPU profiling, Scalene could be very useful.
If you need better visualization of the profiling data, especially for complex processes like 
Markov chains, FunctionTrace or Palanteer might be beneficial.
If you want a balance between ease of use and detailed information, Pyinstrument could be a good option.

To profile for a limited number of iterations, you can modify your main Markov chain function to 
accept a parameter for the number of iterations, and then use the chosen profiler to run this modified 
function. Most of these tools allow you to start and stop profiling programmatically, so you can profile 
specific sections of your code if needed."""

"""Using line_profiler:
line_profiler: This profiler provides line-by-line profiling, which can be very helpful for identifying bottlenecks in specific functions."""

"""from line_profiler import LineProfiler"""

"""   def my_function():
       # Your Markov Chain code here

   profiler = LineProfiler()
   profiler.add_function(my_function)
   profiler.enable_by_count()
   my_function()
   profiler.disable_by_count()
   profiler.print_stats()"""
   
"""Using memory_profiler:
memory_profiler: This profiler is useful for tracking memory usage, which can be a concern for large Markov Chain models."""


"""from memory_profiler import profile

   @profile
   def run_markov_chain():
       # Your Markov Chain code here

   run_markov_chain()"""