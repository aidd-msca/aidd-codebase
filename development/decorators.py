"""This class contains all decorators which can be imported by other classes.

Contributors:
Peter Hartog [ESR1]

"""
# TODO: lazy setting of __init__ variables


def value_error_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypeError:
            print(f"{func.__name__} only takes numbers as the argument")

    return inner_function
