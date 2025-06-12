from .chain import MarkovChain
from falcomchain.partition import Partition
import random
from typing import Union, Callable, List, Any
from tqdm import tqdm
import math


class SingleMetricOptimizer:
    """
    SingleMetricOptimizer represents the class of algorithms / chains that optimize plans with
    respect to a single metric.  An instance of this class encapsulates the following state
    information:
        * the dual graph and updaters via the initial partition,
        * the constraints new proposals are subject to,
        * the metric over which to optimize,
        * and whether or not to seek maximal or minimal values of the metric.

    The SingleMetricOptimizer class implements the following common methods of optimization:
        * Simulated Annealing
        * Tilted Runs

    Both during and after a optimization run, the class properties `best_part` and `best_score`
    represent the optimal partition / corresponding score value observed.  Note that these
    properties do NOT persist across multiple optimization runs, as they are reset each time an
    optimization run is invoked.
    """

    def __init__(
        self,
        proposal: Callable[[Partition], Partition],
        constraints: Union[
            Callable[[Partition], bool], List[Callable[[Partition], bool]]
        ],
        initial_state: Partition,
        optimization_metric: Callable[[Partition], Any],
        maximize: bool = True,
        step_indexer: str = "step",
    ):
        """

        :param proposal: Function proposing the next state from the current state.
        :type proposal: Callable
        :param constraints: A function, or lists of functions, determining whether the proposed next
            state is valid (passes all binary constraints). Usually this is a
            :class:`~gerrychain.constraints.Validator` class instance.
        :type constraints: Union[Callable[[Partition], bool], List[Callable[[Partition], bool]]]
        :param initial_state: Initial state of the optimizer.
        :type initial_state: Partition
        :param optimization_metric: The score function with which to optimize over. This should have
            the signature: ``Partition -> 'a`` where 'a is comparable.
        :type optimization_metric: Callable[[Partition], Any]
        :param maximize: Boolean indicating whether to maximize or minimize the function.
            Defaults to True for maximize.
        :type maximize: bool, optional
        :param step_indexer: Name of the updater tracking the partitions step in the chain. If not
            implemented on the partition the constructor creates and adds it. Defaults to "step".
        :type step_indexer: str, optional



        :return: A SingleMetricOptimizer object
        :rtype: SingleMetricOptimizer
        """
        self._initial_part = initial_state
        self._proposal = proposal
        self._constraints = constraints
        self._score = optimization_metric
        self._maximize = maximize
        self._best_part = None
        self._best_score = None
        self._step_indexer = step_indexer

        if self._step_indexer not in self._initial_part.updaters:
            step_updater = lambda p: (
                0 if p.parent is None else p.parent[self._step_indexer] + 1
            )
            self._initial_part.updaters[self._step_indexer] = step_updater

    @property
    def best_part(self) -> Partition:
        """
        Partition object corresponding to best scoring plan observed over the current (or most recent) optimization run.

        :return: Partition object with the best score.
        """
        return self._best_part


    @property
    def best_score(self) -> Any:
        """
        Value of score metric corresponding to best scoring plan observed over the current (or most recent) optimization run.

        :return: Value of the best score.
        """
        return self._best_score


    @property
    def score(self) -> Callable[[Partition], Any]:
        """
        The score function which is being optimized over.

        :return: The score function.
        :rtype: Callable[[Partition], Any]
        """
        return self._score


    def _is_improvement(self, new_score: float, old_score: float) -> bool:
        """
        Helper function defining improvement comparison between scores.  Scores can be any
        comparable type.

        :param new_score: Score of proposed partition.
        :type new_score: float
        :param old_score: Score of previous partition.
        :type old_score: float

        :return: Whether the new score is an improvement over the old score.
        :rtype: bool
        """
        if self._maximize:
            return new_score >= old_score
        else:
            return new_score <= old_score


    def _tilted_acceptance_function(self, p: float) -> Callable[[Partition], bool]:
        """
        Function factory that binds and returns a tilted acceptance function.

        :param p: The probability of accepting a worse score.
        :type p: float

        :return: An acceptance function for tilted chains.
        :rtype: Callable[[Partition], bool]
        """

        def tilted_acceptance_function(part):
            if part.parent is None:
                return True

            part_score = self.score(part)
            prev_score = self.score(part.parent)

            if self._is_improvement(part_score, prev_score):
                return True
            else:
                return random.random() < p

        return tilted_acceptance_function


    def _simulated_annealing_acceptance_function(
        self, beta_function: Callable[[int], float], beta_magnitude: float
    ):
        """
        Function factory that binds and returns a simulated annealing acceptance function.

        :param beta_function: Function (f: t -> beta, where beta is in [0,1]) defining temperature
            over time.  f(t) = 0 the chain is hot and every proposal is accepted.  At f(t) = 1 the
            chain is cold and worse proposal have a low probability of being accepted relative to
            the magnitude of change in score.
        :type beta_function: Callable[[int], float]
        :param beta_magnitude: Scaling parameter for how much to weight changes in score.
        :type beta_magnitude: float

        :return: A acceptance function for simulated annealing runs.
        :rtype: Callable[[Partition], bool]
        """

        def simulated_annealing_acceptance_function(part):
            if part.parent is None:
                return True
            score_delta = self.score(part) - self.score(part.parent)
            beta = beta_function(part[self._step_indexer])
            if self._maximize:
                score_delta *= -1
            return random.random() < math.exp(-beta * beta_magnitude * score_delta)

        return simulated_annealing_acceptance_function


    def simulated_annealing(
        self,
        num_steps: int,
        beta_function: Callable[[int], float],
        beta_magnitude: float = 1,
        with_progress_bar: bool = False,
    ):
        """
        Performs simulated annealing with respect to the class instance's score function.

        :param num_steps: Number of steps to run for.
        :type num_steps: int
        :param beta_function: Function (f: t -> beta, where beta is in [0,1]) defining temperature
            over time.  f(t) = 0 the chain is hot and every proposal is accepted. At f(t) = 1 the
            chain is cold and worse proposal have a low probability of being accepted relative to
            the magnitude of change in score.
        :type beta_function: Callable[[int], float]
        :param beta_magnitude: Scaling parameter for how much to weight changes in score.
            Defaults to 1.
        :type beta_magnitude: float, optional
        :param with_progress_bar: Whether or not to draw tqdm progress bar. Defaults to False.
        :type with_progress_bar: bool, optional

        :return: Partition generator.
        :rtype: Generator[Partition]
        """
        chain = MarkovChain(
            self._proposal,
            self._constraints,
            self._simulated_annealing_acceptance_function(
                beta_function, beta_magnitude
            ),
            self._initial_part,
            num_steps,
        )

        self._best_part = self._initial_part
        self._best_score = self.score(self._best_part)

        chain_generator = tqdm(chain) if with_progress_bar else chain

        for part in chain_generator:
            yield part
            part_score = self.score(part)
            if self._is_improvement(part_score, self._best_score):
                self._best_part = part
                self._best_score = part_score


    def tilted_run(self, num_steps: int, p: float, with_progress_bar: bool = False):
        """
        Performs a tilted run. A chain where the acceptance function always accepts better plans
        and accepts worse plans with some probability `p`.


        :param num_steps: Number of steps to run for.
        :type num_steps: int
        :param p: The probability of accepting a plan with a worse score.
        :type p: float
        :param with_progress_bar: Whether or not to draw tqdm progress bar. Defaults to False.
        :type with_progress_bar: bool, optional

        :return: Partition generator.
        :rtype: Generator[Partition]
        """
        chain = MarkovChain(
            self._proposal,
            self._constraints,
            self._tilted_acceptance_function(p),
            self._initial_part,
            num_steps,
        )

        self._best_part = self._initial_part
        self._best_score = self.score(self._best_part)

        chain_generator = tqdm(chain) if with_progress_bar else chain

        for part in chain_generator:
            yield part
            part_score = self.score(part)

            if self._is_improvement(part_score, self._best_score):
                self._best_part = part
                self._best_score = part_score
