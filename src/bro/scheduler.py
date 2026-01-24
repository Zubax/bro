"""Task scheduler using APScheduler + OpenMemory."""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from bro.memory import Memory
from bro.reasoner import Reasoner, Context

_logger = logging.getLogger(__name__)


class TaskScheduler:
    def __init__(self, memory: Memory, reasoner: Reasoner) -> None:
        self._memory = memory
        self._reasoner = reasoner
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()
        self._load_scheduled_tasks()
        _logger.info("Scheduler started")

    def _load_scheduled_tasks(self) -> None:
        """Load scheduled tasks from memory on startup."""
        results = self._memory.recall("all scheduled tasks", ["procedural", "scheduled"])
        _logger.info(f"Loaded scheduled tasks from memory: {results}")

        # Parse and re-add tasks: format is "task_id: <id> | prompt: <prompt> | cron: <cron>"
        if not results or "No memories found" in results:
            return

        for line in results.split("\n"):
            if "task_id:" in line and "prompt:" in line and "cron:" in line:
                try:
                    parts = line.split("|")
                    task_id = parts[0].split("task_id:")[1].strip()
                    task_prompt = parts[1].split("prompt:")[1].strip()
                    cron = parts[2].split("cron:")[1].strip()

                    # Check if this task was cancelled
                    cancel_check = self._memory.recall(
                        f"cancelled task {task_id}", ["procedural", "scheduled", "cancelled"]
                    )
                    if cancel_check and f"CANCELLED: {task_id}" in cancel_check:
                        _logger.info(f"Skipping cancelled task: {task_id}")
                        continue

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
                    _logger.info(f"Restored scheduled task: {task_id}")
                except Exception as e:
                    _logger.error(f"Failed to restore task from line '{line}': {e}")

    def _run_scheduled_task(self, task_prompt: str, task_id: str) -> None:
        """Run a scheduled task silently (no user notification)."""
        _logger.info(f"Running scheduled task '{task_id}': {task_prompt}")
        self._reasoner.task(Context(prompt=task_prompt, files=[]), scheduled=True)

    def schedule(self, task_prompt: str, cron: str, task_id: str) -> str:
        """Schedule a task with cron syntax (e.g., '0 9 * * *' for 9am daily)."""
        try:
            # Store in memory
            self._memory.remember(
                f"task_id: {task_id} | prompt: {task_prompt} | cron: {cron}", ["procedural", "scheduled", task_id]
            )

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
            _logger.info(f"Scheduled: {task_id} - {task_prompt} ({cron})")
            return f"Successfully scheduled task '{task_id}'"
        except Exception as e:
            _logger.error(f"Failed to schedule {task_id}: {e}")
            return f"Failed to schedule task: {e}"

    def cancel(self, task_id: str) -> str:
        """Cancel a scheduled task."""
        try:
            # Remove from APScheduler
            self._scheduler.remove_job(task_id)

            # Mark as cancelled in memory (can't delete from OpenMemory)
            self._memory.remember(f"CANCELLED: {task_id}", ["procedural", "scheduled", "cancelled", task_id])

            _logger.info(f"Cancelled: {task_id}")
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
