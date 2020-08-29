import importlib
import os

def import_optional(name, envvarname, fallback=None):
    """Attempt to import the specified module.
    Either returns the module, or None.

    See https://github.com/pandas-dev/pandas/blob/master/pandas/compat/_optional.py
    """
    use_module = os.environ.get('CHEBPY_USE_'+envvarname.upper(), '1')
    if use_module.lower() in ['true', '1']:
        try:
            return importlib.import_module(name)
        except ImportError:
            pass
    if fallback is not None:
        return importlib.import_module(fallback)
    else:
        return None
