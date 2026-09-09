import yaml


class Dotdict(dict):
    """Dictionary with attribute-style access."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, data=None):
        super().__init__()
        for key, value in (data or {}).items():
            if isinstance(value, dict):
                value = Dotdict(value)
            self[key] = value


class HParam(Dotdict):
    def __init__(self, filename):
        with open(filename, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        super().__init__(data)
