import logging
from tqdm import tqdm


def simple_parallel(
    input_list, function, max_cpu=16, timeout=4000, max_retries=3, use_ray:
    bool = False, spawn=False
):
    """Simple parallelization.

    Use map async and retries in case we get odd stalling behavior.

    input_list: Input list to op on
    function: Fn to apply
    max_cpu: Num cpus
    timeout: Length of timeout
    max_retries: Num times to retry this
    use_ray
    spawn=True

    """
    if use_ray:
        import ray

        @ray.remote
        def ray_func(x):
            return function(x)

        return ray.get([ray_func.remote(x) for x in input_list])

    import multiprocess.context as ctx
    from multiprocess.context import TimeoutError
    from pathos import multiprocessing as mp
    if spawn:
        ctx._force_start_method('spawn')

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)
    results = pool.map(function, input_list)
    pool.close()
    pool.join()
    return results


def chunked_parallel(
    input_list,
    function,
    chunks=100,
    max_cpu=16,
    timeout=4000,
    max_retries=3,
    use_ray=False,
    spawn=False,
    desc=None
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
        desc: Description for progress bar (optional)
    """
    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    # Process chunks with progress bar
    if desc:
        # Use parallel processing with progress tracking
        import multiprocess.context as ctx
        from pathos import multiprocessing as mp
        if spawn:
            ctx._force_start_method('spawn')
        
        cpus = min(mp.cpu_count(), max_cpu)
        pool = mp.Pool(processes=cpus)
        
        # Process chunks in parallel with progress bar
        list_outputs = []
        results = []
        for chunk in chunked_list:
            results.append(pool.apply_async(batch_func, (chunk,)))
        
        # Collect results with progress bar
        for result in tqdm(results, desc=desc, total=len(results)):
            list_outputs.append(result.get(timeout=timeout))
        
        pool.close()
        pool.join()
    else:
        list_outputs = simple_parallel(
            chunked_list,
            batch_func,
            max_cpu=max_cpu,
            timeout=timeout,
            max_retries=max_retries,
            use_ray=use_ray,
            spawn=spawn
        )
    # Unroll
    full_output = [j for i in list_outputs for j in i]

    return full_output
