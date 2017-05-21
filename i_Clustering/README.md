# Clustering
Implemented `KMeans`

## Visualization

![KMeans](https://cdn.rawgit.com/carefree0910/Resources/65ca0533/KMeans.gif)

## Example
```python
from Util.Util import DataUtil
from i_Clustering.KMeans import KMeans

x, y = DataUtil.gen_two_clusters()      # Generate two clusters
k_means = KMeans()
k_means.fit(x)                          # Train KMeans (n_clusters: 2)
k_means.visualize2d(x, y, dense=400, extra=k_means["centers"])
                                        # Visualize result (2d)
                                        # Rendering KMeans centers by setting 'extra=k_means["centers"]')
```

### Result
![KMeans](http://oph1aen4o.bkt.clouddn.com/17-5-21/90066307-file_1495346822594_e0ef.png)