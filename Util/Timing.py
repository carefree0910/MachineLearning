import time
import wrapt


class Timing:
    timings = {}
    enabled = False

    def __init__(self, enabled=True):
        Timing.enabled = enabled

    def __str__(self):
        return "Timing"

    __repr__ = __str__

    @classmethod
    def timeit(cls, level=0, func_name=None, cls_name=None, prefix="[Method] "):
        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            if not cls.enabled:
                return func(*args, **kwargs)
            if instance is not None:
                instance_name = "{:>18s}".format(instance.__class__.__name__)
            else:
                instance_name = " " * 18 if cls_name is None else "{:>18s}".format(cls_name)
            _prefix = "{:>26s}".format(prefix)
            try:
                _func_name = "{:>28}".format(func.__name__ if func_name is None else func_name)
            except AttributeError:
                str_func = str(func)
                _at_idx = str_func.rfind("at")
                _dot_idx = str_func.rfind(".", None, _at_idx)
                _func_name = "{:>28}".format(str_func[_dot_idx+1:_at_idx-1])
            _name = instance_name + _prefix + _func_name
            _t = time.time()
            rs = func(*args, **kwargs)
            _t = time.time() - _t
            try:
                cls.timings[_name]["timing"] += _t
                cls.timings[_name]["call_time"] += 1
            except KeyError:
                cls.timings[_name] = {
                    "level": level,
                    "timing": _t,
                    "call_time": 1
                }
            return rs
        return wrapper

    @classmethod
    def show_timing_log(cls, level=2):
        print()
        print("=" * 110 + "\n" + "Timing log\n" + "-" * 110)
        if cls.timings:
            for key in sorted(cls.timings.keys()):
                timing_info = cls.timings[key]
                if level >= timing_info["level"]:
                    print("{:<42s} :  {:12.7} s (Call Time: {:6d})".format(
                        key, timing_info["timing"], timing_info["call_time"]))
        print("-" * 110)

    @classmethod
    def disable(cls):
        cls.enabled = False

if __name__ == '__main__':
    class Test:
        timing = Timing()

        def __init__(self, rate):
            self.rate = rate

        @timing.timeit()
        def test(self, cost=0.1, epoch=3):
            for _ in range(epoch):
                self._test(cost * self.rate)

        @timing.timeit(prefix="[Core] ")
        def _test(self, cost):
            time.sleep(cost)

    class Test1(Test):
        def __init__(self):
            Test.__init__(self, 1)

    class Test2(Test):
        def __init__(self):
            Test.__init__(self, 2)

    class Test3(Test):
        def __init__(self):
            Test.__init__(self, 3)

    test1 = Test1()
    test2 = Test2()
    test3 = Test3()
    test1.test()
    test2.test()
    test3.test()
    test1.timing.show_timing_log()
