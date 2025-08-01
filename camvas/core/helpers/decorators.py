import functools
import inspect
import pathlib


def check_output_dir(func):
    """
    Decorator that ensures the output_dir parameter exists before calling the function.
    Creates the directory if it doesn't exist using pathlib.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature to find output_dir parameter
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Extract output_dir from bound arguments
        if 'output_dir' in bound_args.arguments:
            output_dir = pathlib.Path(bound_args.arguments['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

        return func(*args, **kwargs)

    return wrapper
