# SVM
Implemented `Perceptron`, `Kernel Perceptron`, `LinearSVM` & `SVM`

Implemented **Tensorflow** backend for `LinearSVM` & `SVM`

## Visualization

### Perceptron
![Perceptron on Two Clusters](http://oph1aen4o.bkt.clouddn.com/17-5-20/97196555-file_1495275689415_15d56.gif)

![Perceptron on Two Clusters](http://oph1aen4o.bkt.clouddn.com/17-5-20/62314176-file_1495275694416_47dd.gif)

### LinearSVM
![LinearSVM on Two Clusters](http://oph1aen4o.bkt.clouddn.com/17-5-20/53018892-file_1495275689219_1493b.gif)

![LinearSVM on Two Clusters](http://oph1aen4o.bkt.clouddn.com/17-5-20/97544282-file_1495275694176_6a5.gif)

### TFLinearSVM
![TFLinearSVM on Two Clusters](http://oph1aen4o.bkt.clouddn.com/17-5-20/40345342-file_1495275689828_2c1a.gif)

![TFLinearSVM on Two Clusters](http://oph1aen4o.bkt.clouddn.com/17-5-20/29078272-file_1495275694854_1549d.gif)

### Kernel Perceptron
![Kernel Perceptron on Spiral](http://oph1aen4o.bkt.clouddn.com/17-5-20/29584743-file_1495275688867_17b9d.gif)

![Kernel Perceptron on Spiral](http://oph1aen4o.bkt.clouddn.com/17-5-20/58516048-file_1495275693829_5922.gif)

### SVM
![SVM on Spiral](http://oph1aen4o.bkt.clouddn.com/17-5-20/60620731-file_1495275689578_1d97.gif)

![SVM on Spiral](http://oph1aen4o.bkt.clouddn.com/17-5-20/64162775-file_1495275694615_7489.gif)

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
![SVM on Spiral](http://i1.piimg.com/567571/c1131052c5659373.png)