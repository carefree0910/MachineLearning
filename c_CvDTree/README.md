# Decision Tree
Implemented **ID3**, **C4.5** & **CART**. Capable for dealing with relatively 'raw' data

## Visualization
+ Dataset comes from UCI: [Mushroom dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom)

### ID3
![ID3](http://oph1aen4o.bkt.clouddn.com/18-1-30/99814955.jpg)

### C4.5
![C4.5](http://oph1aen4o.bkt.clouddn.com/18-1-30/48437221.jpg)

### CART
![CART](http://oph1aen4o.bkt.clouddn.com/18-1-30/32419745.jpg)

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
