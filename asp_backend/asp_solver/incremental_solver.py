"""Incremental ASP (clingo) solver for APE"""

import logging
import os
from pathlib import Path
import timeit
from typing import FrozenSet, List, Set, cast

import clingo

from ape_to_asp.read_config import SolverConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


ASP_ENCODING_FILES = [
    'aux.lp',           # rules transforming and completing input facts
    'bind_n.lp',        # [[bind]]_n constraints
    'conf_n.lp',        # [[conf]]_n constraints
    'constraints.lp',   # constraints from nl templates and config flags
    'goal.lp',          # bind workflow output at last step
    'in_out_n.lp',      # [[in]]_n and [[out]]_n
    'io.lp',            # #show statements
    'tax_t_op_n.lp',    # [[tax_t]]_n and [[tax_op]]_n rules
    'workflow.lp',      # generator for use_tool, in, out and bind
]


class InstanceLoadedError(RuntimeError):
    """Raised when loading workflow configs / instances more than once."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class NoInstanceError(RuntimeError):
    """Raised when trying to start solving with no instance loaded."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class SolverNotInitialized(RuntimeError):
    """Raised when trying to check or add steps without initializing the solver."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class IncASPSolver:
    """Incremental ASP (clingo) solver for APE"""

    def __init__(
        self,
        encoding_path: str,
        heuristics: bool = False,
        _logger: logging.Logger = logger,
    ) -> None:
        self._logger = _logger
        self._ctl = clingo.Control(
            ['0']
            + (['--heuristic=Domain'] if heuristics else [])
        )
        # load solution encoding
        for path in ASP_ENCODING_FILES:
            enc_path = os.path.join(encoding_path, path)
            if not os.path.exists(enc_path):
                raise FileNotFoundError(f'Encoding not found: {enc_path}')
            self._logger.debug('Loading encoding: %s', enc_path.split("/")[-1])
            self._ctl.load(enc_path)
        self._instance_loaded = False
        self._step = -1
        self.models: Set[FrozenSet[str]] = set()


    def load_instance_from_files(
        self,
        tax_path: str,
        tool_mode_path: str,
        constraint_path: str,
        io_path: str,
        domain_asp_path: str='',
        workflow_asp_path: str='',
    ) -> None:
        """Load instance from asp encoding files.

        Args:
            tax_path (str): Path to taxonomy encoding
            tool_mode_path (str): Path to tool mode io encoding
            constraint_path (str): Path to wf constraint encoding
            io_path (str): Path to wf io

        Raises:
            InstanceLoadedError: Solver already loaded an instance.
            FileNotFoundError: An encoding file was not found.
        """
        if self._instance_loaded:
            raise InstanceLoadedError()
        instance_paths = [
            tax_path,
            tool_mode_path,
            constraint_path,
            io_path,
        ] + ([] if domain_asp_path == '' else [domain_asp_path]) \
            + ([] if workflow_asp_path == '' else [workflow_asp_path])
        for path in instance_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f'Instance encoding not found: {path}')
        # load instance
        for path in instance_paths:
            self._logger.debug('Loading encoding: %s', path.split("/")[-1])
            self._ctl.load(path)
        self._instance_loaded = True


    def load_instance_from_symbols(
        self,
        tax_symbols: List[clingo.Symbol],
        tool_mode_symbols: List[clingo.Symbol],
        constraint_symbols: List[clingo.Symbol],
        io_symbols: List[clingo.Symbol],
    ) -> None:
        """Load instance from lists of symbols and add them via clingo backend.

        Raises:
            InstanceLoadedError: Solver already loaded an instance.
        """
        if self._instance_loaded:
            raise InstanceLoadedError()
        with self._ctl.backend() as backend:
            for symbol in tax_symbols + tool_mode_symbols + constraint_symbols + io_symbols:
                new_atom = backend.add_atom(symbol)
                backend.add_rule([new_atom])
        self._instance_loaded = True


    def _ground_init(self) -> None:
        if not self._instance_loaded:
            raise NoInstanceError()
        self._ctl.ground([('base', []), ('check', [clingo.Number(0)])])
        self._ctl.assign_external(
            clingo.Function('query', [clingo.Number(0)]),
            True,
        )
        self._step = 0


    def _retrieve_step_models(self, num_models: int=1) -> int:
        if self._step < 0:
            raise SolverNotInitialized()
        if num_models < 1:
            raise ValueError('num_models must be at least 1.')
        with cast(clingo.SolveHandle, self._ctl.solve(yield_=True)) as handle:
            if not handle.get().satisfiable:
                return 0
            old_num_models = len(self.models)
            while len(self.models) < num_models:
                if (model := handle.model()):
                    self.models.add(frozenset(str(symbol) for symbol in model.symbols(shown=True)))
                    handle.resume()
                else:
                    return len(self.models) - old_num_models
            return num_models


    def _add_step(self) -> None:
        if self._step < 0:
            raise SolverNotInitialized()
        self._ctl.release_external(clingo.Function('query', [clingo.Number(self._step)]))
        self._ctl.cleanup()
        self._ctl.ground([
            ('step', [clingo.Number(self._step+1)]),
            ('check', [clingo.Number(self._step+1)]),
        ])
        self._ctl.assign_external(
            clingo.Function('query', [clingo.Number(self._step+1)]),
            True,
        )
        self._step += 1


    def solve_instance(
        self,
        config: SolverConfig,
    ) -> None:
        """Runs incremental solving process until a solution has been found
        or the step/time limit has been reached.

        Args:
            config (SolverConfig): Dataclass containing step and timelimits.

        Raises:
            RuntimeError: The solver was already used.
        """
        if len(self.models) > 0:
            raise RuntimeError('Solver instance not new.')
        init_start_time = timeit.default_timer()
        self._ground_init()
        for _ in range(config.solution_length_min):
            self._add_step()
        init_end_time = timeit.default_timer()
        self._logger.debug('Solver init time: %.2fs', init_end_time - init_start_time)
        start_time = timeit.default_timer()
        self._retrieve_step_models(config.solutions)
        self._logger.info('Step %d, Models %d', self._step, len(self.models))
        while (
            len(self.models) < config.solutions
            and self._step <= config.solution_length_max
            and timeit.default_timer() - start_time <= config.timeout
        ):
            self._add_step()
            self._retrieve_step_models(config.solutions)
            self._logger.info('Step %d, Models %d', self._step, len(self.models))


    def save_models(self, config: SolverConfig) -> None:
        """Saves current set of models in specified path.

        Args:
            config (SolverConfig): Config containing the path.
        """
        solutions_path = Path(config.solutions_dir_path)
        solutions_path.mkdir(parents=True, exist_ok=True)
        for model_ix, model in enumerate(self.models, start=1):
            with (
                solutions_path / f'model_{model_ix}.lp'
            ).open('w', encoding='utf-8') as model_file:
                for symbol in model:
                    model_file.write(f'{symbol}.\n')
