# falcomchain

A Python library for **capacitated facility location** via Markov Chain Monte Carlo (MCMC) sampling on graph partitions.

falcomchain implements a hierarchical and capacitated ReCom algorithm that simultaneously partitions a dual graph into hierarchical service districts, locate facilities in districts, and allocates expert teams to facilities, while satisfying capacity-demand balance and user-choice constraints such as budget.

> **Status:** Pre-publication. The library is under active development. The Chicago healthcare dataset used for testing is not part of the paper. A new experiment will be designed separately.

---

## What it does

Given a graph where nodes are geographic units (e.g., census blocks) with population attributes, falcomchain:

1. Partitions the graph into contiguous districts using a capacitated spanning tree algorithm
2. Assigns a number of doctor-nurse teams to each district based on population and capacity constraints
3. Runs an MCMC chain over the space of valid partitions using hierarchical ReCom proposals
4. Tracks objectives (compactness, cut edges, radius deviation) to guide or evaluate the chain

---

## Installation

```bash
git clone https://github.com/kirtisoglu/Allocation-of-Primary-Care-Centers-in-Chicago
cd Allocation-of-Primary-Care-Centers-in-Chicago
pip install -e .
```

Requires Python 3.12+.

---

## Quick start

```python
from falcomchain.graph import Graph
from falcomchain.partition import Partition

# Load your graph (must have 'population' node attribute)
graph = Graph.from_file("my_graph.geojson")

# Create an initial partition
partition = Partition.from_random_assignment(
    graph=graph,
    pop_target=1500,
    epsilon=0.1,
    capacity_level=2,
)

# Run the chain
from falcomchain.markovchain import MarkovChain, hierarchical_recom, always_accept
from functools import partial

proposal = partial(hierarchical_recom, pop_target=1500, epsilon=0.1)
chain = MarkovChain(proposal=proposal, accept=always_accept, initial_state=partition, total_steps=100)

for state in chain:
    print(state.step, len(state.parts))
```

---

## Repository structure

See [docs/structure.md](docs/structure.md) for a detailed breakdown of every module.

---

## Algorithm

See [docs/algorithm.md](docs/algorithm.md) for an explanation of ReCom, hierarchical ReCom, capacitated tree partitioning, and the MCMC framework.

---

## Data

See [docs/data.md](docs/data.md) for a description of the data files used for testing and what a new experiment will require.

---

## For AI agents

See [docs/agent_guide.md](docs/agent_guide.md) for a navigation guide to the codebase intended for LLM assistants.

---

## License

MIT License. See [LICENSE.txt](LICENSE.txt).
