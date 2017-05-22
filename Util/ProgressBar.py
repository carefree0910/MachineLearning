import time


class ProgressBar:
    def __init__(self, min_value=0, max_value=None, min_refresh_period=0.5, width=30, name="", start=True):
        self._min, self._max = min_value, max_value
        self._task_length = int(max_value - min_value) if (
            min_value is not None and max_value is not None
        ) else None
        self._counter = min_value
        self._min_period = min_refresh_period
        self._bar_width = int(width)
        self._bar_name = " " if not name else " # {:^12s} # ".format(name)
        self._terminated = False
        self._started = False
        self._ended = False
        self._current = 0
        self._clock = 0
        self._cost = 0
        if start:
            self.start()

    def _flush(self):
        if self._ended:
            return False
        if not self._started:
            print("Progress bar not started yet.")
            return False
        if self._terminated:
            if self._counter == self._min:
                self._counter = self._min + 1
            self._cost = time.time() - self._clock
            tmp_hour = int(self._cost / 3600)
            tmp_min = int((self._cost - tmp_hour * 3600) / 60)
            tmp_sec = self._cost % 60
            tmp_avg = self._cost / (self._counter - self._min)
            tmp_avg_hour = int(tmp_avg / 3600)
            tmp_avg_min = int((tmp_avg - tmp_avg_hour * 3600) / 60)
            tmp_avg_sec = tmp_avg % 60
            print(
                "\r" +
                "##{}({:d} : {:d} -> {:d}) Task Finished. "
                "Time Cost: {:3d} h {:3d} min {:6.4} s; Average: {:3d} h {:3d} min {:6.4} s ".format(
                    self._bar_name, self._task_length, self._min, self._counter - self._min,
                    tmp_hour, tmp_min, tmp_sec, tmp_avg_hour, tmp_avg_min, tmp_avg_sec
                ) + " ##\n", end=""
            )
            self._ended = True
            return True
        if self._counter >= self._max:
            self._terminated = True
            return self._flush()
        if self._counter != self._min and time.time() - self._current <= self._min_period:
            return False
        self._current = time.time()
        self._cost = time.time() - self._clock
        if self._counter > self._min:
            tmp_hour = int(self._cost / 3600)
            tmp_min = int((self._cost - tmp_hour * 3600) / 60)
            tmp_sec = self._cost % 60
            tmp_avg = self._cost / (self._counter - self._min)
            tmp_avg_hour = int(tmp_avg / 3600)
            tmp_avg_min = int((tmp_avg - tmp_avg_hour * 3600) / 60)
            tmp_avg_sec = tmp_avg % 60
        else:
            print()
            tmp_hour = 0
            tmp_min = 0
            tmp_sec = 0
            tmp_avg_hour = 0
            tmp_avg_min = 0
            tmp_avg_sec = 0
        passed = int(self._counter * self._bar_width / self._max)
        print(
            "\r" + "##{}[".format(
                self._bar_name
            ) + "-" * passed + " " * (self._bar_width - passed) + "] : {} / {}".format(
                self._counter, self._max
            ) + " ##  Time Cost: {:3d} h {:3d} min {:6.4} s; Average: {:3d} h {:3d} min {:6.4} s ".format(
                tmp_hour, tmp_min, tmp_sec, tmp_avg_hour, tmp_avg_min, tmp_avg_sec
            ) if self._counter != self._min else "##{}Progress bar initialized  ##".format(
                self._bar_name
            ), end=""
        )
        return True

    def set_min(self, min_val):
        if self._max is not None:
            if self._max <= min_val:
                print("Target min_val: {} is larger than current max_val: {}".format(min_val, self._max))
                return
            self._task_length = self._max - min_val
        self._counter = self._min = min_val

    def set_max(self, max_val):
        if self._min is not None:
            if self._min >= max_val:
                print("Target max_val: {} is smaller than current min_val: {}".format(max_val, self._min))
                return
            self._task_length = max_val - self._min
        self._max = max_val

    def update(self, new_value=None):
        if new_value is None:
            new_value = self._counter + 1
        if new_value != self._min:
            self._counter = self._max if new_value >= self._max else int(new_value)
            return self._flush()

    def start(self):
        if self._task_length is None:
            print("Error: Progress bar not initialized properly.")
            return
        self._current = self._clock = time.time()
        self._started = True
        self._flush()

    def terminate(self):
        self._terminated = True
        self._flush()

if __name__ == '__main__':

    def task(cost=0.25, epoch=3, name="", _sub_task=None):
        def _sub():
            bar = ProgressBar(max_value=epoch, name=name)
            for _ in range(epoch):
                time.sleep(cost)
                if _sub_task is not None:
                    _sub_task()
                bar.update()

        return _sub


    task(name="Task1", _sub_task=task(
        name="Task2", _sub_task=task(
            name="Task3")))()
