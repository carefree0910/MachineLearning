from abc import ABCMeta


from Util.Timing import Timing


class TimingMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        timing = Timing()

        for _name, _value in attr.items():
            if "__" in _name or "timing" in _name or "estimate" in _name:
                continue
            _str_val = str(_value)
            if "<" not in _str_val and ">" not in _str_val:
                continue
            if _str_val.find("function") >= 0 or _str_val.find("staticmethod") >= 0 or _str_val.find("property") >= 0:
                attr[_name] = timing.timeit(level=2)(_value)

        def __str__(self):
            try:
                return self.name
            except AttributeError:
                return name

        def __repr__(self):
            return str(self)

        def show_timing_log(self, level=2):
            getattr(self, name + "Timing").show_timing_log(level)

        for key, value in locals().items():
            if str(value).find("function") >= 0 or str(value).find("property"):
                attr[key] = value

        return type(name, bases, attr)


class SubClassTimingMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        timing = Timing()
        for _name, _value in attr.items():
            if "__" in _name or "timing" in _name or "estimate" in _name:
                continue
            _str_val = str(_value)
            if "<" not in _str_val and ">" not in _str_val:
                continue
            if _str_val.find("function") >= 0 or _str_val.find("staticmethod") >= 0 or _str_val.find("property") >= 0:
                attr[_name] = timing.timeit(level=2)(_value)
        return type(name, bases, attr)


class SKCompatibleMeta(ABCMeta):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]

        def __init__(self, *_args, **_kwargs):
            for base in bases:
                base.__init__(self, *_args, **_kwargs)
        attr["__init__"] = __init__
        return type(name, bases, attr)
