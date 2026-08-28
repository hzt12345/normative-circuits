"""Run independent dataset cells while preserving a truthful process status."""

import traceback


class DatasetBatchError(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        detail = ", ".join(
            f"{name}: {message}" for name, message in failures.items()
        )
        super().__init__(f"Dataset batch failed: {detail}")


def run_dataset_batch(datasets, run_one):
    """Attempt every dataset, then fail if any cell raised an exception."""
    results = {}
    failures = {}
    for dataset in datasets:
        try:
            results[dataset] = run_one(dataset)
        except Exception as exc:
            traceback.print_exc()
            failures[dataset] = f"{type(exc).__name__}: {exc}"
    if failures:
        raise DatasetBatchError(failures)
    return results
