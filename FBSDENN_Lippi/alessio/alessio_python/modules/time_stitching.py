"""Compatibility entrypoint for the modular TensorFlow 2 implementation.

The maintained logic now lives across:
- config.py
- models.py
- train.py
- utils.py
- visualization.py
- main.py

Running this file preserves the old entrypoint while delegating to the updated
modular stack.
"""

from main import main


if __name__ == "__main__":
    main()
