from publang.utils.string import format_string_with_variables


def test_format_string_with_variables():
    # Test case with no variables
    assert format_string_with_variables("Hello, World!") == "Hello, World!"

    # Test case with one variable
    assert format_string_with_variables(
        "Hello, {name}!", name="Alice") == "Hello, Alice!"

    # Test case with multiple variables
    assert format_string_with_variables(
        "Hello, {name}! My favorite color is {color}.", name="Bob", color="blue") == "Hello, Bob! My favorite color is blue."