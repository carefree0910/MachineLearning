# Decision Tree
Implemented **ID3**, **C4.5** & **CART**. Capable for dealing with relatively 'raw' data

## Visualization
+ Dataset comes from UCI: [Mushroom dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom)

### ID3
![image](http://i1.piimg.com/567571/b202b2dfd1394757.png)

### C4.5
![image](http://i1.piimg.com/567571/d64bffa200033d00.png)

### CART
![image](http://i1.piimg.com/567571/330a93ad355c0a05.png)

## Example
```python
from Util.Util import DataUtil
from c_CvDTree.Tree import CartTree

x, y = DataUtil.gen_xor(one_hot=False)  # Get xor dataset. Notice that y should not be one-hot
tree = CartTree()                       # Use Cart Tree for example
tree.fit(x, y, train_only=True)         # Use all dataset for training
tree.view()                             # View trained Cart Tree in console
tree.evaluate(x, y)                     # Print out accuracy 
tree.visualize2d(x, y)                  # Visualize result (2d)
tree.visualize()                        # Visualize Cart Tree itself
tree.show_timing_log()                  # Show timing log
```