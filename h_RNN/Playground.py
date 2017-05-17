import numpy as np


class RNN1:
    def __init__(self, u, v, w):
        self._u, self._v, self._w = np.asarray(u), np.asarray(v), np.asarray(w)
        self._states = None

    def activate(self, x):
        return x

    def transform(self, x):
        return x

    def run(self, x):
        output = []
        x = np.atleast_2d(x)
        self._states = np.zeros([len(x)+1, self._u.shape[0]])
        for t, xt in enumerate(x):
            self._states[t] = self.activate(
                self._u.dot(xt) + self._w.dot(self._states[t-1])
            )
            output.append(self.transform(
                self._v.dot(self._states[t]))
            )
        return np.array(output)


class RNN2(RNN1):
    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def transform(self, x):
        safe_exp = np.exp(x - np.max(x))
        return safe_exp / np.sum(safe_exp)

    def bptt(self, x, y):
        x, y, T = np.asarray(x), np.asarray(y), len(y)
        o = self.run(x)
        dis = o - y
        dv = dis.T.dot(self._states[:-1])
        du = np.zeros_like(self._u)
        dw = np.zeros_like(self._w)
        for t in range(T-1, -1, -1):
            ds = self._v.T.dot(dis[t])
            for bptt_step in range(t, max(-1, t-10), -1):
                du += np.outer(ds, x[bptt_step])
                dw += np.outer(ds, self._states[bptt_step-1])
                st = self._states[bptt_step-1]
                ds = self._w.T.dot(ds) * st * (1 - st)
        return du, dv, dw

if __name__ == '__main__':
    _T = 5
    rnn = RNN1(np.eye(_T), np.eye(_T), np.eye(_T) * 2)
    print(rnn.run(np.eye(_T)))
