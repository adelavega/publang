import concurrent.futures
import tqdm
import json


def parallelize_inputs(func):
    """Decorator to parallelize the extraction process over texts."""

    def wrapper(inputs, *args, output_path=None, **kwargs):
        # Get the number of workers (pop)
        num_workers = kwargs.pop("num_workers", 1)

        # If only one, run in serial mode
        if not isinstance(inputs, list):
            num_workers = 1
            inputs = [inputs]

        if num_workers != 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as exc:
                inputs = [exc.submit(func, i, *args, **kwargs) for i in inputs]

        # Loop through the inputs
        results = []
        for input in tqdm.tqdm(inputs):
            # Serial mode
            if num_workers == 1:
                results.append(func(input, *args, **kwargs))
            else:
                results.append(input.result())

            if len(results) % 10 == 0:
                if output_path is not None:
                    json.dump(results, open(output_path, "w"))
        else:
            # If only one result, delist
            if len(results) == 1:
                results = results[0]
            if output_path is not None:
                json.dump(results, open(output_path, "w"))
        return results

    return wrapper
