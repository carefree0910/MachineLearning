# Naive Bayes
Naive Bayes algorithms implemented with
+ `np.bincount` for `MultinomialNB`
+ `np.exp` for `GaussianNB`

`MultinomialNB` + `GaussianNB` = `MergedNB`

## Visualization
+ Dataset comes from UCI: [Bank Marketing dataset](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

![Categorical Features](http://oph1aen4o.bkt.clouddn.com/18-1-30/53941771.jpg)

![Numerical Features](http://oph1aen4o.bkt.clouddn.com/18-1-30/53966960.jpg)

## Example
```python
from Util.Util import DataUtil
from b_NaiveBayes.Vectorized.GaussianNB import GaussianNB

x, y = DataUtil.gen_xor(one_hot=False)  # Get xor dataset. Notice that y should not be one-hot
nb = GaussianNB()
nb.fit(x, y)                            # Train GaussianNB
nb.evaluate(x, y)                       # Print out accuracy
nb.visualize2d(x, y)                    # Visualize result (2d)
nb.visualize()                          # Visualize distribution
nb.show_timing_log()                    # Show timing log
```
