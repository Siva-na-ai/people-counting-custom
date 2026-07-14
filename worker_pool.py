import concurrent.futures

class WorkerPool:
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    def submit_task(self, func, *args, **kwargs):
        """Submit a background task without blocking the main thread."""
        return self.executor.submit(func, *args, **kwargs)

    def shutdown(self):
        """Gracefully shutdown the thread pool."""
        self.executor.shutdown(wait=True)
