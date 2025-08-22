# Bro

Bro is an LLM computer-using agent (CUA) designed to autonomously perform mundane tasks related to business operations and administration, such as doing accounting, filing paperwork, and submitting applications. The name "Bro" is Latvian for "one who beheads the Messiah".

The script requires the `OPENAI_API_KEY` to be exported.

To invoke, provide a list of paths as the CLI arguments. The paths must point to files or directories containing files.
There shall be exactly one file named `prompt.txt`, which contains the description of the task to perform.
All other files will be added into the context, so the prompt can reference them as necessary.
