from collections import Counter as Count


class Counter:

    def __init__(self, arr, sample_weights=None):
        if sample_weights is None:
            self._counter = Count(arr)
        else:
            self._counter = {}
            sw_len = len(sample_weights)
            for elem, w in zip(arr, sample_weights):
                if elem not in self._counter:
                    self._counter[elem] = w * sw_len
                else:
                    self._counter[elem] += w * sw_len

    def values(self):
        return self._counter.values()

    def __getitem__(self, item):
        return self._counter[item]

if __name__ == '__main__':
    data = [1, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    weights1 = [0.1] * 10
    weights2 = [0.05, 0.15] * 5
    counter1 = Counter(data)
    counter2 = Counter(data, weights1)
    counter3 = Counter(data, weights2)
    print(counter1.values())
    print(counter2.values())
    print(counter3.values())
