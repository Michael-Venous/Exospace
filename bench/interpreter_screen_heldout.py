"""Run the frozen new labels/requests through screenshot-only Jev."""
from bench import interpreter_heldout  # installs the independent case/label set
from bench.interpreter_screen import main

if __name__ == '__main__':
    main()
