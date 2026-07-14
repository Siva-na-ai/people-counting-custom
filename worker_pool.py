import queue
import threading
import logging
from typing import Callable, Any

logger = logging.getLogger("worker_pool")

class WorkerPool:
    """
    Background worker queue running database updates, offline gallery maintenance,
    and event logging sequentially on a separate thread to keep real-time inference non-blocked.
    """
    def __init__(self):
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.running = True
        self.worker_thread.start()

    def _run_loop(self):
        logger.info("[WorkerPool] Background thread started.")
        while self.running:
            try:
                # Fetch task with timeout to check running state periodically
                func, args, kwargs = self.task_queue.get(timeout=1.0)
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"[WorkerPool] Exception running task {func.__name__}: {e}")
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[WorkerPool] Fatal error in loop: {e}")

    def submit_task(self, func: Callable[..., Any], *args, **kwargs):
        """
        Pushes a task to the background queue.
        """
        if not self.running:
            logger.warning("[WorkerPool] Task submitted but worker is stopped.")
            return
        self.task_queue.put((func, args, kwargs))

    def stop(self):
        """
        Stops the worker thread cleanly.
        """
        self.running = False
        # Block until thread finishes current task
        self.worker_thread.join(timeout=3.0)
        logger.info("[WorkerPool] Background thread stopped.")
        
    def pending_tasks_count(self) -> int:
        return self.task_queue.qsize()
