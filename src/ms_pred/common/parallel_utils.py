import logging
from tqdm import tqdm
import multiprocess.context as ctx
ctx._force_start_method('spawn')

def simple_parallel(
    input_list, function, max_cpu=16, timeout=4000, max_retries=3, use_ray: bool = False
):
    """Simple parallelization.

    Use map async and retries in case we get odd stalling behavior.

    input_list: Input list to op on
    function: Fn to apply
    max_cpu: Num cpus
    timeout: Length of timeout
    max_retries: Num times to retry this
    use_ray

    """
    if use_ray:
        import ray

        @ray.remote
        def ray_func(x):
            return function(x)

        return ray.get([ray_func.remote(x) for x in input_list])

    from multiprocess.context import TimeoutError
    from pathos import multiprocessing as mp

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)
    results = pool.map(function, input_list)
    pool.close()
    pool.join()
    return results

# # If parallel with default multi-processing, this class is needed to pass the function call object
# class batch_func:
#     def __init__(self, func, args=None, kwargs=None):
#         self.func = func
#         if args is not None:
#             self.args = args
#         else:
#             self.args = []
#         if kwargs is not None:
#             self.kwargs = kwargs
#         else:
#             self.kwargs = {}
#
#     def __call__(self, list_inputs):
#         outputs = []
#         for i in list_inputs:
#             outputs.append(self.func(i, *self.args, **self.kwargs))
#         return outputs


def chunked_parallel(
    input_list,
    function,
    chunks=100,
    max_cpu=16,
    timeout=4000,
    max_retries=3,
    use_ray=False,
):
    """chunked_parallel.

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of hcunks
        max_cpu: Max num cpus
        timeout: Length of timeout
        max_retries: Num times to retry this
        use_ray
    """
    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    if list_len == 0:
        raise ValueError('Empty list to process!')
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    from pathos import multiprocessing as mp
    cpus = min(mp.cpu_count(), max_cpu)
    with mp.ProcessPool(processes=cpus) as pool:
        list_outputs = list(tqdm(pool.imap(batch_func, chunked_list), total=len(chunked_list)))

    # import multiprocessing as mp
    # cpus = min(mp.cpu_count(), max_cpu)
    # with mp.get_context("spawn").Pool(cpus) as pool:
    #     func_obj = batch_func(function, args, kwargs)
    #     list_outputs = list(tqdm(pool.imap(func_obj, chunked_list), total=num_chunks))

    # # Parallel without tqdm
    # list_outputs = simple_parallel(
    #     chunked_list,
    #     batch_func,
    #     max_cpu=max_cpu,
    #     timeout=timeout,
    #     max_retries=max_retries,
    #     use_ray=use_ray,
    # )

    # Unroll
    full_output = [j for i in list_outputs for j in i]

    return full_output
