# SVM
Implemented `Perceptron`, `Kernel Perceptron`, `LinearSVM` & `SVM`

Implemented `Tensorflow` & `PyTorch` backend for `LinearSVM` & `SVM`

## Visualization

### Perceptron
![Perceptron on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Lines/Perceptron.gif)

![Perceptron on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Backgrounds/Perceptron.gif)

### LinearSVM
![LinearSVM on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Lines/LinearSVM.gif)

![LinearSVM on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Backgrounds/LinearSVM.gif)

### TFLinearSVM
![TFLinearSVM on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Lines/TFLinearSVM.gif)

![TFLinearSVM on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Backgrounds/TFLinearSVM.gif)

### TorchLinearSVM
![TorchLinearSVM on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/cbd5675e/Lines/TorchLinearSVM.gif)

![TorchLinearSVM on Two Clusters](https://cdn.rawgit.com/carefree0910/Resources/cbd5675e/Backgrounds/TorchLinearSVM.gif)

### Kernel Perceptron
![Kernel Perceptron on Spiral](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Lines/KP.gif)

![Kernel Perceptron on Spiral](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Backgrounds/KP.gif)

### SVM
![SVM on Spiral](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Lines/SVM.gif)

![SVM on Spiral](https://cdn.rawgit.com/carefree0910/Resources/d269faeb/Backgrounds/SVM.gif)

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
