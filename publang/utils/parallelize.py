import concurrent.futures
import tqdm
import json


def _save_results(results, output_path, every=None):
    """Save the results to a file."""
    if output_path is not None:
        if every is None:
            json.dump(results, open(output_path, "w"))
        elif len(results) % every == 0:
            json.dump(results, open(output_path, "w"))
    return results


def parallelize_inputs(func):
    """Decorator to parallelize the extraction process over texts."""

    def wrapper(inputs, *args, output_path=None, **kwargs):
        # Get the number of workers (pop)
        num_workers = kwargs.pop("num_workers", 1)

        # If only one, run in serial mode
        if not isinstance(inputs, list):
            result = func(inputs, *args, **kwargs)
            _save_results(result, output_path)
            return result

        # Run in parallel mode
        if num_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exc:
                futures = [
                    exc.submit(func, i, *args, **kwargs) for i in inputs
                ]
                results = []
                for r in tqdm.tqdm(concurrent.futures.as_completed(futures),
                                   total=len(inputs)):
                    results.append(r.result())
                    _save_results(results, output_path, every=10)
        else:
            results = []
            for input in tqdm.tqdm(inputs):
                results.append(func(input, *args, **kwargs))
                _save_results(results, output_path, every=10)

        _save_results(results, output_path)
        return results

    return wrapper
