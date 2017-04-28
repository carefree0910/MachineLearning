[1]:https://github.com/carefree0910/ImageRecognition/tree/master
[2]:https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip "Inception-v3"

# Image Recognition

Dependency: numpy, matplotlib, Tensorflow, cv2(For visualization)

A Stand-alone version for this project can be found [here][1]

+ Inception-v3 model for this project can be downloaded [here][2]
+ Extract the zipped file and put Inception-v3 model (which should be renamed from 'tensorflow_inception_graph.pb' to 'Model.pb') to 'Models/Extractor/v3' folder
+ Put your training set **FOLDERS** into '_Data' folder, please use English names for your folders to ensure that cv2 works correctly
    + Each folder name should be treated as the 'label' of the pictures contained in the folder
+ Put your test set **PICTURES** into 'Test' folder
    + If possible, put a **ONE-HOT** answer naming '_answer.npy' into 'Test' folder as well for better visualization
    + If you don't want to struggle for these, just leave 'Test' folder empty (Reference the Notice below)
+ Run 'Main.py'!
    + Run 'Main.py' in PyCharm will be perfectly correct. If you want to run it by double-clicking, some import statements may need to be modified
    + You may want a [Stand-alone version][1] for this project, where you can run it by double-clicking 'Main.py' without modifying any import statement!

## Notice That:
+ If 'Test' folder remains empty when the program is running, `min(196, 0.2 * n_data)`pictures will be **MOVED** from '_Data' folder to 'Test' folder if 'gen_test' FLAG is True
    + An '_answer.npy' ndarray will also be generated automatically!
+ After processing all images in '_Data' folder, a '_Cache' folder which contains 'features.npy' and 'labels.npy' (shuffled) will be generated
    + If you want to train on new dataset, '_Cache' folder should be deleted
    + You can train your own classifier using 'features.npy' and 'labels.npy'
+ After the program is done, a Predictor will be stored in 'Models/Predictors/v3' folder. If you want to train on new dataset, this folder should be deleted

--args:

parser.add_argument(

    "--gen_test",
    type=bool,
    default=True,
    help="Whether generate test images"
)

parser.add_argument(

    "--images_dir",
    type=str,
    default="Test",
    help="Path to test set"
)

parser.add_argument(

    "--extract_only",
    type=bool,
    default=False,
    help="Whether extract features only"
)

parser.add_argument(

    "--visualize_only",
    type=bool,
    default=False,
    help="Whether visualize only"
)

parser.add_argument(

    "--overview",
    type=bool,
    default=True,
    help="Whether use cv2 to overview"
)

## Visualization

![image](http://i1.piimg.com/567571/663724d1c2d0c997.png)

![image](http://i1.piimg.com/567571/f253925c8122775a.png)

*(Not so elegant, but (maybe) better than nothing...)*
