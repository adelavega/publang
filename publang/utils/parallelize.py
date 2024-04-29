import concurrent.futures
import tqdm


def parallelize_inputs(func):
    """Decorator to parallelize the extraction process over texts."""

    def wrapper(inputs, *args, **kwargs):
        # Get the number of workers (pop)
        num_workers = kwargs.pop("num_workers", 1)

        # If only one, run in serial mode
        if not isinstance(inputs, list):
            result = func(inputs, *args, **kwargs)

            return result

        # Run in parallel mode
        if num_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exc:
                futures = [
                    exc.submit(func, i, *args, **kwargs) for i in inputs
                ]
                results = [
                    r.result() for r in tqdm.tqdm(futures, total=len(futures))
                ]

        else:
            results = [
                func(i, *args, **kwargs) for i in tqdm.tqdm(inputs)
            ]
        return results

    return wrapper
