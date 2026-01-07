I will create a `.cursorrules` file in the project root and update `CLAUDE.md` to permanently record your environment and tool preferences.

**Plan:**
1.  **Create `.cursorrules`**: This is a standard configuration file used by AI coding assistants to define project-specific behavior. I will add the following content:
    ```markdown
    ## Environment & Tools
    - **Shell**: Use `fish` shell syntax for all terminal commands.
    - **Python Manager**: ALWAYS use `uv` for package management.
    - **Install**: `uv add <package>` (or `uv add --dev <package>`)
    - **Run**: `uv run python script.py`
    - **Prohibited**: Do NOT use `pip install` or `python -m venv`.
    ```
2.  **Update `CLAUDE.md`**: I will also prepend these rules to the top of `CLAUDE.md` to ensure they are immediately visible in the project's primary context file.

This ensures that in future sessions, I (and other AI agents) will see these rules and follow your preferred workflow.