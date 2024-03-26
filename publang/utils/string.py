import re


def format_string_with_variables(string: str, **kwargs: str) -> str:
    # Find all possible variables in the string
    possible_variables = set(re.findall(r"{(\w+)}", string))

    # Find all provided variables in the kwargs dictionary
    provided_variables = set(kwargs.keys())

    # Check that all provided variables are in the possible variables
    if not provided_variables.issubset(possible_variables):
        raise ValueError(f"Provided variables {provided_variables} are not in the possible variables {possible_variables}.")

    # Format the string with the provided variables
    return string.format(**kwargs)