import time

def timeit(func):
    """
    Decorator to measure the runtime of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' executed in {runtime:.6f} seconds.")
        return result

    return wrapper