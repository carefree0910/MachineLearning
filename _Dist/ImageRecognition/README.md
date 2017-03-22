# 图像分类

将“世界杯”、“地球”、……等文件夹放在 _Data 文件夹中、并将想要测试的图片放在 Test 文件夹中后、运行 Main.py 即可
+ 程序会将 _Data 文件夹中的文件夹重命名为 “0000”、“0001”、……，这是因为 Tensorflow 框架貌似不认非 ASCII 码的路径
+ 在处理完一遍 _Data 文件夹中的所有图片后，程序会在 _Data 文件夹中生成一个 _Cache 文件夹并把图片处理结果储存在其中；如果想要在新的训练集上训练的话、就需要把它和 Models/Predictors 里的 v3 文件夹删掉
    + 这样做是为了让用户可以尝试使用 v3 提取出来的特征来训练自己想训练的分类器
+ 如果 Test 文件夹为空的话、程序会自动从 _Data 文件夹中随机**剪切** 200 张图片到 Test 文件夹里

程序中只有 Main.py 能够运行、其它程序都是开发过程所用；它们的依赖文件太大、所以我没有放进来

程序依赖：numpy、sklearn、matplotlib、Tensorflow

命令行参数说明：

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

    "--model",
    type=str,
    default="v3",
    help="Model used to extract & predict"
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
    help="Whether overview"
)