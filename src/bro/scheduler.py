"""Task scheduler using APScheduler + OpenMemory."""

import logging
from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]
from bro.memory import Memory
from bro.reasoner import Reasoner, Context

_logger = logging.getLogger(__name__)


class TaskScheduler:
    def __init__(self, memory: Memory, reasoner: Reasoner) -> None:
        self._memory = memory
        self._reasoner = reasoner
        self._scheduler = BackgroundScheduler()
        self._task_memory_ids: dict[str, str] = {}  # task_id -> memory_id mapping
        self._scheduler.start()
        self._load_scheduled_tasks()
        _logger.info("Scheduler started")

    def _load_scheduled_tasks(self) -> None:
        """Load scheduled tasks from memory on startup."""
        # Query returns all matching memories with their IDs
        results = self._memory._memory.query("scheduled tasks", filters={"tags": ["scheduled"]})
        _logger.info(f"Found {len(results)} scheduled task memories")

        for memory_entry in results:
            content = memory_entry.get("content", "")
            memory_id = memory_entry.get("id", "")

            if "task_id:" in content and "prompt:" in content and "cron:" in content:
                try:
                    parts = content.split("|")
                    task_id = parts[0].split("task_id:")[1].strip()
                    task_prompt = parts[1].split("prompt:")[1].strip()
                    cron = parts[2].split("cron:")[1].strip()

                    # Store the memory ID for later deletion
                    self._task_memory_ids[task_id] = memory_id

                    minute, hour, day, month, day_of_week = cron.split()
                    self._scheduler.add_job(
                        lambda p=task_prompt, tid=task_id: self._run_scheduled_task(p, tid),
                        "cron",
                        minute=minute,
                        hour=hour,
                        day=day,
                        month=month,
                        day_of_week=day_of_week,
                        id=task_id,
                        replace_existing=True,
                    )
                    _logger.info(f"Restored scheduled task: {task_id} (memory: {memory_id})")
                except Exception as e:
                    _logger.error(f"Failed to restore task from memory '{content}': {e}")

    def _run_scheduled_task(self, task_prompt: str, task_id: str) -> None:
        """Run a scheduled task silently (no user notification)."""
        _logger.info(f"Running scheduled task '{task_id}': {task_prompt}")
        self._reasoner.task(Context(prompt=task_prompt, files=[]), scheduled=True)

    def schedule(self, task_prompt: str, cron: str, task_id: str) -> str:
        """Schedule a task with cron syntax (e.g., '0 9 * * *' for 9am daily)."""
        try:
            # Store in memory and save the memory ID
            memory_id = self._memory.remember(
                f"task_id: {task_id} | prompt: {task_prompt} | cron: {cron}", ["procedural", "scheduled", task_id]
            )
            self._task_memory_ids[task_id] = memory_id

            # Add to APScheduler
            minute, hour, day, month, day_of_week = cron.split()
            self._scheduler.add_job(
                lambda: self._run_scheduled_task(task_prompt, task_id),
                "cron",
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
                id=task_id,
                replace_existing=True,
            )
            _logger.info(f"Scheduled: {task_id} - {task_prompt} ({cron}) [memory: {memory_id}]")
            return f"Successfully scheduled task '{task_id}'"
        except Exception as e:
            _logger.error(f"Failed to schedule {task_id}: {e}")
            return f"Failed to schedule task: {e}"

    def cancel(self, task_id: str) -> str:
        """Cancel a scheduled task."""
        try:
            # Remove from APScheduler
            self._scheduler.remove_job(task_id)

            # Delete from memory
            memory_id = self._task_memory_ids.get(task_id)
            if memory_id:
                self._memory.forget(memory_id)
                del self._task_memory_ids[task_id]
                _logger.info(f"Cancelled: {task_id} (deleted memory {memory_id})")
            else:
                _logger.warning(f"Cancelled {task_id} but no memory ID found")

            return f"Successfully cancelled task '{task_id}'"
        except Exception as e:
            _logger.error(f"Failed to cancel {task_id}: {e}")
            return f"Failed to cancel task: {e}"

    def list_tasks(self) -> str:
        """List all scheduled tasks."""
        jobs = self._scheduler.get_jobs()
        if not jobs:
            return "No scheduled tasks"

        result = "Scheduled tasks:\n"
        for job in jobs:
            result += f"- {job.id}: next run at {job.next_run_time}\n"
        return result
