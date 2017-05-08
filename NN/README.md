# NN
Implemented **Neural Network** & **Convolutional Neural Network** with `numpy` & `Tensorflow`

## Visualization
![Neural Network on Spiral](https://cdn.rawgit.com/carefree0910/Resources/641e8228/NN.gif)

## Examples

### Visualize Network
```python
from NN.Basic.Networks import *
from Util.Util import DataUtil

nn = NNDist()

timing = Timing()
timing_level = 1

x, y = DataUtil.gen_spiral(50, 3, 3, 2.5)

nn.build([x.shape[1], 6, 6, 6, y.shape[1]])  # Build a neural network on the fly (With ReLU + Cross Entropy)
nn.optimizer = "Adam"                        # Use Adam algorithms for training 
nn.preview()                                 # Preview network structure    
nn.feed_timing(timing)
nn.fit(x, y, verbose=1, record_period=4, epoch=1000, train_only=True,
       draw_detailed_network=True, show_animation=True, make_mp4=True)
                                             # Visualize network & make an mp4 file                    
nn.draw_results()                            # Draw results (Training curve & loss curve)
nn.visualize2d()                             # Visualize result (2d)

timing.show_timing_log(timing_level)         # Show timing log
```

#### Result
See the GIF above

### Mnist
```python
from NN.NN import *
from Util.Util import DataUtil

timing = Timing()
timing_level = 1

x, y = DataUtil.get_dataset("mnist", "../../_Data/mnist.txt", quantized=True, one_hot=True)
                                             # Get tiny mnist dataset
x = x.reshape(len(x), 1, 28, 28)             # Reshape x to 4d array

nn = NNDist()

# # Neural Network
# nn.add("ReLU", (x.shape[1], 24))
# nn.add("ReLU", (24, ))
# nn.add("CrossEntropy", (y.shape[1], ))

# Convolutional Neural Network
nn.add("ConvReLU", (x.shape[1:], (32, 3, 3)))
nn.add("ConvReLU", ((32, 3, 3),))
nn.add("MaxPool", ((3, 3),), 2)
nn.add("ConvNorm")
nn.add("ConvDrop")
nn.add("ConvReLU", ((64, 3, 3),), std=0.01)
nn.add("ConvReLU", ((64, 3, 3),), std=0.01)
nn.add("AvgPool", ((3, 3),), 2)
nn.add("ConvNorm")
nn.add("ConvDrop")
nn.add("ConvReLU", ((32, 3, 3),))
nn.add("ConvReLU", ((32, 3, 3),))
nn.add("AvgPool", ((3, 3),), 2)
nn.add("ReLU", (512,))
nn.add("Identical", (64,))
nn.add("Normalize", activation="ReLU")
nn.add("Dropout")
nn.add("CrossEntropy", (y.shape[1],))

nn.optimizer = "Adam"
nn.preview()                     
nn.fit(x, y, verbose=2, do_log=True, show_loss=True)
                                             # Train network 
nn.draw_results()                            # Draw results (Training curve & loss curve)

timing.show_timing_log(timing_level)         # Show timing log
```
