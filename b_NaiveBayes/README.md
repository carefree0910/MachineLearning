# Naive Bayes
Naive Bayes algorithms implemented with
+ `np.bincount` for `MultinomialNB`
+ `np.exp` for `GaussianNB`

`MultinomialNB` + `GaussianNB` = `MergedNB`

## Visualization
+ Dataset comes from UCI: [Bank Marketing dataset](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

![Discrete Features](http://i4.buimg.com/567571/79ea5cb079b16a72.png)

![Continuous Features](http://i2.muimg.com/567571/f267b1a607e4c95e.png)

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