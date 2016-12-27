import time
import wrapt


class Timing:
    _timings = {}
    _enabled = False

    def __init__(self, enabled=False):
        Timing._enabled = enabled

    @staticmethod
    def timeit(level=0, name=None, cls_name=None, prefix="[Function] "):
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            if not Timing._enabled:
                return func(*args, **kwargs)
            if instance is not None:
                instance_name = "{:>18s}".format(str(instance))
            else:
                instance_name = " " * 18 if cls_name is None else "{:>18s}".format(cls_name)
            _prefix = "{:>26s}".format(prefix)
            func_name = "{:>28}".format(func.__name__ if name is None else name)
            _name = instance_name + _prefix + func_name
            _t = time.time()
            rs = func(*args, **kwargs)
            _t = time.time() - _t
            try:
                Timing._timings[_name]["timing"] += _t
                Timing._timings[_name]["call_time"] += 1
            except KeyError:
                Timing._timings[_name] = {
                    "level": level,
                    "timing": _t,
                    "call_time": 1
                }
            return rs
        return wrapper

    @property
    def timings(self):
        return self._timings

    def show_timing_log(self, level):
        print()
        print("=" * 110 + "\n" + "Timing log\n" + "-" * 110)
        if not self.timings:
            print("None")
        else:
            for key in sorted(self.timings.keys()):
                timing_info = self.timings[key]
                if level >= timing_info["level"]:
                    print("{:<42s} :  {:12.7} s (Call Time: {:6d})".format(
                        key, timing_info["timing"], timing_info["call_time"]))
        print("-" * 110)
