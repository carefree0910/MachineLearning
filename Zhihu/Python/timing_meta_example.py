import time
import wrapt


class Timing:
    _timings = {}
    _enabled = False

    def __init__(self, enabled=True):
        Timing._enabled = enabled

    def __str__(self):
        return "Timing"

    __repr__ = __str__

    @staticmethod
    def timeit(level=0, name=None, cls_name=None, prefix="[Method] "):
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

    def show_timing_log(self, level=2):
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


class TimingBase:
    def feed_timing(self, timing):
        pass

    def show_timing_log(self, level=2):
        pass


class TimingMeta(type):
    def __new__(mcs, *args, **kwargs):
        name, bases, attr = args[:3]
        try:
            _timing = attr[name + "Timing"]
        except KeyError:
            _timing = Timing()
            attr[name + "Timing"] = _timing

        for _name, _value in attr.items():
            if "__" in _name or "timing" in _name or "estimate" in _name:
                continue
            _str_val = str(_value)
            if "<" not in _str_val and ">" not in _str_val:
                continue
            if _str_val.find("function") >= 0 or _str_val.find("staticmethod") >= 0 or _str_val.find("property") >= 0:
                attr[_name] = _timing.timeit(level=2)(_value)

        def feed_timing(self, timing):
            setattr(self, name + "Timing", timing)

        def show_timing_log(self, level=2):
            getattr(self, name + "Timing").show_timing_log(level)

        attr["feed_timing"] = feed_timing
        attr["show_timing_log"] = show_timing_log

        return type(name, bases, attr)


class Test(TimingBase, metaclass=TimingMeta):

    @staticmethod
    def test1():
        for i in range(10 ** 7):
            pass

    @staticmethod
    def test2():
        for i in range(10 ** 7):
            _ = 1

    @staticmethod
    def test3():
        for i in range(10 ** 7):
            pass
            pass

    @staticmethod
    def test4():
        for i in range(10 ** 7):
            _ = 1
            _ = 2

    @staticmethod
    def test5():
        for i in range(10 ** 7):
            _ = 1
            _ = 1

if __name__ == '__main__':
    test = Test()
    test.test1()
    test.test2()
    test.test3()
    test.test4()
    test.test5()
    test.show_timing_log()
