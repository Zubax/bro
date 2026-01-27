from pathlib import Path

# Note: if Bro is run from the home directory, the local and global brodirs are the same.
BRODIR_GLOBAL = Path().home() / "bro/"
BRODIR_LOCAL = Path("bro/")

BRODIR_GLOBAL.mkdir(parents=True, exist_ok=True)
BRODIR_LOCAL.mkdir(parents=True, exist_ok=True)

# Specific brofiles.
USER_SYSTEM_PROMPT_FILE = BRODIR_GLOBAL / "system_prompt.txt"
EMAIL_WORKFLOW_PROMPT_FILE = BRODIR_GLOBAL / "email_workflow_prompt.txt"
INVOICE_CREATION_PROMPT_FILE = BRODIR_GLOBAL / "invoice_creation_prompt.txt"
SNAPSHOT_FILE = BRODIR_LOCAL / "state.bro.json"
LOG_FILE = BRODIR_LOCAL / "bro.log"
LOG_DB = BRODIR_LOCAL / "bro.db"
MEMORY_DB = BRODIR_LOCAL / "memory.db"
