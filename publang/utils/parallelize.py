import concurrent.futures
import tqdm
import json


def parallelize_inputs(func):
    """Decorator to parallelize the extraction process over texts."""

    def wrapper(inputs, *args, output_path=None, **kwargs):
        # Get the number of workers (pop)
        num_workers = kwargs.pop("num_workers", 1)

        # If only one text is provided, run the function directly
        if not isinstance(inputs, list):
            return func(inputs, *args, **kwargs)
                
        # If num_workers is 1, run the function directly
        if num_workers == 1:
            results = []
            for i in tqdm.tqdm(inputs):
                results.append(func(i, *args, **kwargs))
                if output_path is not None and len(results) % 10 == 0:
                    json.dump(results, open(output_path, "w"))

            return results

        # Otherwise, parallelize the function
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exc:
            futures = [exc.submit(func, i, *args, **kwargs) for i in inputs]

        results = []
        for future in tqdm.tqdm(futures, total=len(inputs)):
            results.append(future.result())
            assert 0
            # Save every 10 results
            if output_path is not None and len(results) % 10 == 0:
                json.dump(results, open(output_path, "w"))
        return results

    return wrapper
