from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Sequence
import numpy as np

Path = Sequence[int]

class Decoration:
    def __init__(self, path: Path, pre: str, post: str) -> None:
        pass

class Declaration:
    is_fixed: bool
    is_function: bool
    only_root: bool

class Context:
    @staticmethod
    def standard() -> Context:
        pass
    @staticmethod
    def load(filename: str) -> Context:
        pass
    def add_function(self, name: str, fixed: Optional[bool]) -> None:
        pass
    declarations: dict[str, Declaration]

class CnnEmbedding:
    idents: np.ndarray
    index_map: np.ndarray
    positional_encoding: np.ndarray
    rules: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    target: np.ndarray
    mask: np.ndarray

UnrolledEmbedding = tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]

class GraphEmbedding:
    nodes: np.ndarray
    receivers: np.ndarray
    senders: np.ndarray
    n_node: np.ndarray
    n_edge: np.ndarray
    value: np.ndarray
    target: np.ndarray
    mask: np.ndarray

class Symbol:
    # creation
    @staticmethod
    def parse(context: Context, code: str) -> Symbol:
        pass
    @staticmethod
    def variable(ident: str, fixed: bool) -> Symbol:
        pass
    @staticmethod
    def number(value: float) -> Symbol:
        pass
    # visualization
    label: str
    tree: str
    ident: str
    verbose: str
    latex: str
    latex_verbose: str
    depth: int
    childs: Sequence[Symbol]
    size: int
    memory_usage: int
    number_of_embedded_properties: int
    # traversing
    parts_dfs: Sequence[Symbol]
    parts_bfs: Sequence[Symbol]
    parts_bfs_with_path: Sequence[tuple[Path, Symbol]]

    def clone(self) -> Symbol:
        pass
    def at(self, path: Path) -> Symbol:
        pass
    def latex_with_deco(self, decorations: Sequence[Decoration]) -> str:
        pass
    def latex_with_colors(self, colors: Sequence[tuple[str, Path]]) -> str:
        pass
    def pad(self, padding: str, spread: int, depth: int) -> None:
        pass
    def replace_and_pad(
        self, pattern: str, target: str, pad_size: int, pad_symbol: Symbol
    ) -> Symbol:
        pass
    def create_graph_embedding(
        self,
        ident2id: dict[str, int],
        target_size: int,
        fits: Sequence[FitInfo],
        useful: bool,
        use_additional_features: bool,
    ) -> GraphEmbedding:
        pass
    def create_embedding(
        self,
        ident2id: dict[str, int],
        padding: int,
        spread: int,
        max_depth: int,
        target_size: str,
        fits: Sequence[FitInfo],
        useful: bool,
        index_map: bool,
        positional_encoding: bool,
        use_additional_features: bool,
    ) -> CnnEmbedding:
        pass
    def embed(
        self,
        ident2index: dict[str, int],
        padding: int,
        spread: int,
        max_depth: int,
        target_size,
        fits: Sequence[FitInfo],
        useful: bool,
        index_map: bool,
        positional_encoding: bool,
        use_additional_features: bool,
    ) -> UnrolledEmbedding:
        pass
    def create_padded(self, adding: str, spread: int, depth: int) -> Symbol:
        pass

class Rule:
    def __init__(self, condition: Symbol, conclusion: Symbol, name: str):
        pass
    @staticmethod
    def parse(context: Context, code: str, name: Optional[str] = None):
        """Just uses the first rule."""
        pass
    def replace_and_pad(
        self, pattern: str, target: str, pad_size: int, pad_symbol: Symbol
    ) -> Rule:
        """Replace each occurrence of `pattern` with `target` and pads the node with `pad_symbol`."""
        pass
    condition: Symbol
    conclusion: Symbol
    name: str
    verbose: str
    latex: str
    latex_verbose: str
    reverse: Rule

class SymbolBuilder:
    def add_level_uniform(self, child_per_arm: int = 2) -> None:
        """Adds childs at each arm uniformly"""
        pass
    def set_level_idents(self, level: int, idents: Sequence[str]) -> None:
        """
        Sets the idents of all symbols of the specified level in the order of traversing
        to the given ident.
        If there are more entries in the list the remaining get ignored.
        """
        pass
    def get_level_idents(self, level: int) -> Sequence[str]:
        pass
    @property
    def symbol(self) -> Symbol:
        pass

class ScenarioProblems:
    def dump(self, filename: str):
        pass
    @staticmethod
    def load(filename: str) -> ScenarioProblems:
        pass
    def add_to_validation(self, rule: Rule):
        pass
    def add_to_training(self, rule: Rule):
        pass
    def add_additional_idents(self, idents: Sequence[str]):
        pass
    @property
    def validation(self) -> Sequence[Rule]:
        pass
    @property
    def training(self) -> Sequence[Rule]:
        pass
    @property
    def all(self) -> Sequence[Rule]:
        pass
    @property
    def additional_idents(self) -> Sequence[str]:
        pass

class Scenario:
    @staticmethod
    def load(filename: str) -> Scenario:
        pass
    rules: Sequence[Rule]
    problems: Optional[ScenarioProblems]
    declarations: Context
    def idents(self, ignore_declaration: bool = True) -> Sequence[str]:
        pass
    tagset_size: int
    def vocab_size(self, ignore_declaration: bool = True) -> int:
        pass
    spread: int

# Statistics
class StepInfo:
    current_latex: str
    value: Optional[float]
    confidence: Optional[float]
    subsequent: Sequence[StepInfo]
    rule_id: int
    path: Path
    top: int
    contributed: bool
    def add_subsequent(self, other: StepInfo) -> None:
        pass
    def __iadd__(self, other: StepInfo) -> StepInfo:
        pass

class TraceStatistics:
    success: bool
    fit_tries: int
    fit_results: int
    trace: StepInfo

class IterationSummary:
    def __init__(
        self,
        fit_results: Optional[int] = None,
        success: Optional[bool] = None,
        max_depth: Optional[int] = None,
        depth_of_solution: Optional[int] = None,
    ) -> None:
        pass
    fit_results: Optional[int]
    success: Optional[bool]
    max_depth: Optional[int]
    max_depth_of_solution: Optional[int]

class ProblemSummary:
    def __init__(
        self,
        name: str,
        success: bool,
        iterations: Optional[Sequence[IterationSummary]] = None,
        initial_latex: Optional[str] = None,
        target_latex: Optional[str] = None,
    ):
        pass
    @property
    def initial_latex(self) -> Optional[str]:
        pass
    @property
    def iterations(self) -> Sequence[IterationSummary]:
        pass
    @property
    def name(self) -> str:
        pass
    @property
    def success(self) -> bool:
        pass

class ProblemStatistics:
    def __init__(self, problem_name: str, target_latex: str) -> None:
        pass
    problem_name: str
    target_latex: str
    @property
    def iterations(self) -> Sequence[TraceStatistics]:
        pass
    def add_iteration(self, trace: TraceStatistics) -> None:
        pass
    def __iadd__(self, other: TraceStatistics) -> ProblemStatistics:
        pass

class SolverStatistics:
    @staticmethod
    def load(filename: str) -> SolverStatistics:
        pass
    def dump(self, filename: str) -> None:
        pass
    def get_problem(self, problem_name: str) -> ProblemStatistics:
        pass
    def add_problem(self, problem: ProblemStatistics) -> None:
        pass
    def create_index(self, progress_reporter: Callable[[float], None]) -> None:
        pass
    @property
    def header(self) -> Sequence[ProblemSummary]:
        pass
    def __iadd__(self, problem: ProblemStatistics) -> SolverStatistics:
        pass

# Sample

class Sample:
    def __init__(
        self, initial: Symbol, fits: Sequence[FitInfo], useful: bool = True
    ) -> None:
        pass
    @property
    def initial(self) -> Symbol:
        pass
    @property
    def fits(self) -> Sequence[FitInfo]:
        pass
    @property
    def useful(self) -> bool:
        pass
    def create_cnn_embedding(
        self,
        ident2index: dict[str, int],
        padding: int,
        spread: int,
        max_depth: int,
        target_size: int,
        index_map: bool,
        positional_encoding: bool,
        use_additional_features: bool,
    ) -> CnnEmbedding:
        pass
    def embed_cnn(
        self,
        ident2index: dict[str, int],
        padding: int,
        spread: int,
        max_depth: int,
        target_size: int,
        index_map: bool,
        positional_encoding: bool,
        use_additional_features: bool,
    ) -> UnrolledEmbedding:
        pass

class Container:
    def add_sample(self, sample: Sample) -> None:
        pass
    @property
    def max_depth(self) -> int:
        pass
    @property
    def max_spread(self) -> int:
        pass
    @property
    def max_size(self) -> int:
        pass
    @property
    def samples(self) -> Sequence[Sample]:
        pass
    @property
    def samples_with_policy(self) -> Sequence[Sample]:
        pass

class SampleSet:
    def add(self, sample: Sample) -> bool:
        pass
    def merge(self, other: SampleSet) -> None:
        pass
    def to_container(self) -> Container:
        pass
    def fill_possibilities(self, rule_mapping: dict[int, Rule]) -> None:
        pass
    @staticmethod
    def from_container(container: Container) -> SampleSet:
        pass
    def keys(self) -> Sequence[str]:
        pass
    def values(self) -> Sequence[Sample]:
        pass
    def items(self) -> Sequence[tuple[str, Sample]]:
        pass
    def __iadd__(self, problem: Sample) -> SampleSet:
        pass
    def __len__(self) -> int:
        pass

class BagMeta:
    @staticmethod
    def from_scenario(scenario: Scenario, ignore_declaration: bool = True) -> BagMeta:
        pass
    @property
    def idents(self) -> Sequence[str]:
        pass
    @property
    def rules(self) -> Sequence[Rule]:
        pass
    @property
    def rule_distribution(self) -> Sequence[tuple[int, int]]:
        pass
    @property
    def value_distribution(self) -> tuple[int, int]:
        pass
    def clear_distributions(self) -> None:
        pass
    def clone_with_distribution(self, samples: Sequence[Sample]) -> BagMeta:
        pass
    def update_distributions(self, samples: Sequence[Sample]) -> None:
        pass

class Bag:
    def __init__(self, rules: Sequence[tuple[Any, Rule]]) -> None:
        pass
    @staticmethod
    def from_scenario(scenario: Scenario, ignore_declaration: bool = True) -> Bag:
        pass
    @staticmethod
    def load(filename: str) -> Bag:
        pass
    def dump(self, filename: str) -> None:
        pass
    @property
    def meta(self) -> BagMeta:
        pass
    @property
    def containers(self) -> Sequence[Container]:
        pass
    def add_container(self, container: Container) -> None:
        pass
    def update_meta(self) -> None:
        pass
    def clear_containers(self) -> None:
        pass

# Trace

class ApplyInfo:
    @property
    def rule(self) -> Rule:
        pass
    @property
    def path(self) -> Path:
        pass
    @property
    def initial(self) -> Symbol:
        pass
    @property
    def deduced(self) -> Symbol:
        pass

class Calculation:
    steps: Sequence[ApplyInfo]

class Meta:
    @property
    def used_idents(self) -> Sequence[str]:
        pass
    @property
    def rules(self) -> Sequence[Rule]:
        pass

class Trace:
    @staticmethod
    def load(filename: str) -> Trace:
        pass
    @property
    def unroll(self) -> Iterator[Calculation]:
        pass
    @property
    def all_steps(self) -> Iterator[ApplyInfo]:
        pass
    @property
    def meta(self) -> Meta:
        pass

# Fitting

class FitInfo:
    def __init__(self, rule_id: int, path: Path, positive: bool):
        pass
    rule: int
    path: Path
    policy: float

class FitMap:
    path: Path
    variable: dict[Symbol, Symbol]

def fit(outer: Symbol, inner: Symbol) -> Sequence[FitMap]:
    del outer, inner

def fit_at(outer: Symbol, inner: Symbol, path: Path) -> Optional[FitMap]:
    del outer, inner, path

VariableCreator = Callable[[], Symbol]

def apply(
    mapping: FitMap, variable_creator: VariableCreator, orig: Symbol, conclusion: Symbol
) -> Symbol:
    pass

def fit_and_apply(
    variable_creator: VariableCreator, orig: Symbol, rule: Rule
) -> Sequence[tuple[Symbol, FitMap]]:
    del variable_creator, orig, rule

def fit_at_and_apply(
    variable_creator: VariableCreator, orig: Symbol, rule: Rule, path: Path
) -> Optional[tuple[Symbol, FitMap]]:
    del variable_creator, orig, rule, path
