# SVM
Implemented `Perceptron`, `Kernel Perceptron`, `LinearSVM` & `SVM`

Implemented **Tensorflow** backend for `LinearSVM` & `SVM`

## Example
```python
from Util.Util import DataUtil
from e_SVM.SVM import SVM

x, y = DataUtil.gen_spiral(20, 4, 2, 2, one_hot=False)
y[y == 0] = -1                          # Get spiral dataset, Notice that y should be 1 or -1

svm = SVM()
svm.fit(x, y, kernel="poly", p=12)      # Train SVM (kernel: poly, degree: 12)
svm.evaluate(x, y)                      # Print out accuracy
svm.visualize2d(x, y, padding=0.1, dense=400, emphasize=svm["alpha"] > 0)
                                        # Visualize result (2d) (emphasized support vector)
```

### Result
![image](http://i1.piimg.com/567571/c1131052c5659373.png)