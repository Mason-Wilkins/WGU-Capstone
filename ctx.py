# ctx.py
from contextlib import contextmanager
from flask import current_app
from functools import wraps

@contextmanager
def ensure_app_context():
    """Use existing app context if present; otherwise push a temporary one."""
    try:
        _ = current_app.name  # raises RuntimeError if no active app ctx
        yield
    except RuntimeError:
        # Lazily import to avoid circular imports
        from server import app
        with app.app_context():
            yield


def with_app_context(func):
    """Decorator: ensure a Flask app context for the wrapped function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with ensure_app_context():
            return func(*args, **kwargs)
    return wrapper