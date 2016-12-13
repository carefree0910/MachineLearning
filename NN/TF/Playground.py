def evaluate(self, x, y, metrics=None):
    # x = NN._transfer_x(np.array(x))
    if metrics is None:
        metrics = self._metrics
    else:
        for i in range(len(metrics) - 1, -1, -1):
            metric = metrics[i]
            if isinstance(metric, str):
                if metric not in self._available_metrics:
                    metrics.pop(i)
                else:
                    metrics[i] = self._available_metrics[metric]
    logs, y_pred = [], self._get_prediction(x, verbose=2, out_of_sess=True)
    for metric in metrics:
        logs.append(metric(y, y_pred))
    return logs


def _append_log(self, x, y, name, get_loss=True, out_of_sess=False):
    y_pred = self._get_prediction(x, name, out_of_sess=out_of_sess)
    for i, metric in enumerate(self._metrics):
        self._logs[name][i].append(metric(y, y_pred))
    if get_loss:
        if not out_of_sess:
            self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())
        else:
            with self._sess.as_default():
                self._logs[name][-1].append(self._layers[-1].calculate(y, y_pred).eval())