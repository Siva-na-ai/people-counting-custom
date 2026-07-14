import logging
import threading
import queue

logger = logging.getLogger(__name__)

class WorkerPool:
    def __init__(self, num_workers=2):
        self.queue = queue.Queue()
        self.workers = []
        self.num_workers = num_workers
        self.running = True
        
        for i in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)
            
        logger.info(f"Started WorkerPool with {num_workers} workers")

    def _worker_loop(self):
        while self.running:
            try:
                task, args, kwargs = self.queue.get(timeout=1.0)
                try:
                    task(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Worker task failed: {e}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

    def submit(self, task, *args, **kwargs):
        self.queue.put((task, args, kwargs))

    def shutdown(self):
        self.running = False
        for t in self.workers:
            t.join(timeout=2.0)
