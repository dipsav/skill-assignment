"""Import-isolated surge benchmark package; do not copy these functions into VNS.py.

Use: import surge_tools; surge_tools.Surge_benchmarking(...)
"""
from pathlib import Path
import inspect

from . import surge_simulation
from . import surge_case_analysis
from . import surge_benchmarking
from .surge_benchmarking import Surge_benchmarking, iter_benchmark_results
from .surge_case_analysis import (
    Surge_phase_costs,
    Surge_phase_costs_from_case,
    aggregate_case_records,
    load_case_json,
    save_case_json,
)


def verify_imports() -> dict[str, str]:
    """Validate the engine API and return the actual source-file locations."""
    engine = surge_simulation
    required = ("simulation", "simulation_batch_run", "print_surge_report")
    missing = [name for name in required if not callable(getattr(engine, name, None))]
    if missing or not hasattr(engine, "PHASES"):
        raise ImportError(
            f"Incomplete surge engine at {engine.__file__!r}; "
            f"required callable(s) missing: {missing}; PHASES must also exist. "
            "Extract the complete surge_tools folder."
        )
    options = {"numberSurgeRequests", "queueReportEveryArrivals", "recoveryHoldArrivals"}
    absent = options.difference(inspect.signature(engine.simulation_batch_run).parameters)
    if absent:
        raise ImportError(f"Incompatible surge engine; missing options: {sorted(absent)}")
    if surge_case_analysis._surge_engine is not engine:
        raise ImportError("The case wrapper is bound to a different engine.")
    if surge_benchmarking.Surge_phase_costs is not surge_case_analysis.Surge_phase_costs:
        raise ImportError("The benchmark runner is bound to a different case wrapper.")
    return {
        "package_file": str(Path(__file__).resolve()),
        "benchmark_file": str(Path(surge_benchmarking.__file__).resolve()),
        "wrapper_file": str(Path(surge_case_analysis.__file__).resolve()),
        "engine_file": str(Path(engine.__file__).resolve()),
        "benchmark_function_module": Surge_benchmarking.__module__,
        "batch_function_module": engine.simulation_batch_run.__module__,
    }


__all__ = [
    "Surge_benchmarking", "Surge_phase_costs", "Surge_phase_costs_from_case",
    "iter_benchmark_results", "aggregate_case_records", "load_case_json",
    "save_case_json", "verify_imports", "surge_simulation",
]
