from pathlib import Path

def get_root_dir(start_path: Path | None = None) -> Path:
    """
    Return the project root directory, defined as the first parent folder
    (starting from `start_path`) that contains a `pyproject.toml`.

    If `start_path` is None, it starts from the current file's directory.
    Raises FileNotFoundError if no pyproject.toml is found.
    """
    if start_path is None:
        # If used inside a script/module
        try:
            start_path = Path(__file__).resolve()
        except NameError:
            # Fallback for environments without __file__ (e.g. REPL, notebook)
            start_path = Path.cwd().resolve()

    # If a file path is given, search from its parent directory
    if start_path.is_file():
        current = start_path.parent
    else:
        current = start_path

    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file():
            return directory

    raise FileNotFoundError("Could not find 'pyproject.toml' in any parent directory.")

