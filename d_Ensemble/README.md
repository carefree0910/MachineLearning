# Ensemble learning
Implemented `Random Forest` & `AdaBoost`

## Example
```python
from Util.Util import DataUtil
from d_Ensemble.RandomForest import RandomForest

x, y = DataUtil.gen_spiral(size=20, n=4, n_class=2, one_hot=False) 
y[y == 0] = -1                          # Get spiral dataset, Notice that y should be 1 or -1
rf = RandomForest()
rf.fit(x, y)                            # Train Random Forest (Using Cart Tree as default)
rf.evaluate(x, y)                       # Print out accuracy 
rf.visualize2d(x, y)                    # Visualize result (2d)
rf.show_timing_log()                    # Show timing log
```

### Result
![Cart on Spiral](http://i4.buimg.com/567571/fd50b722d988b1dd.png)