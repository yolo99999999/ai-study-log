PyTorch基础

一、张量tensor
   是一种与数组和矩阵非常相似的专用数据结构。 在PyTorch中，使用张量编码模型的输入和输出，以及模型参数；类似于NumPy的ndarray，不同之处在于张量可以在GPU或其他硬件加速器上运行；和 NumPy 数组通常可以共享相同的底层内存，从而消除复制数据的需求。
① 直接从数据中生成 torch.tensor
② 由 NumPy 数组创建 torch.from_numpy(np_array)，由tensor变np数组 tensor.numpy()
③ 新张量保留参数张量的性质（形状、数据类型），除非被显式覆盖。如 x_ones = torch.ones_like(x_data) 
④ 用 shape 定义张量维数。
    如 shape = (2,3,)
        rand_tensor = torch.rand(shape)  返回一个2组3维的张量
⑤ 默认情况下，张量是在CPU上创建的。但可在检查加速器的可用性后，显式地用方法将张量移动到加速器。
    if torch.accelerator.is_available():
        tensor = tensor.to(torch.accelerator.current_accelerator())
⑥ 索引和切片  
    如  tensor[:,1] = 0   # 选取所有行，第二列直接原地赋值0，不返回新tensor
⑦ 连接张量 torch.cat
    如  t1 = torch.cat([tensor, tensor, tensor], dim=1)  # dim=1 在列方向上
⑧ tensor.T 支持任意维度，总是交换最后两个轴
⑨ 算术运算
	@ / matmul → 矩阵乘，维度可以 ≥ 2，支持 batch
	* / mul → 元素乘，Broadcast 规则一样生效
	带out= 的写法，如 y3 = torch.rand_like(y1)
	                             torch.matmul(tensor, tensor.T, out=y3)   #运算结果直接给到y3。
⑩ 单元张量 item()  把 只含单个元素 的 0-D 张量（scalar tensor）转换成 Python 原生数字（float / int / bool），并返回该数字。即把 单元素张量 变成 普通数字 的快捷方式，方便打印、记录、回传非张量接口。

二、数据集、数据处理
1、存储样本 DataLoader 及其对应标签 datasets
from torch.utils.data import cataLoader  
from torchvision import datasets

2、PyTorch 提供领域特定的库，如 TorchText、TorchVision 和 TorchAudio。
	其中 TorchVision 模块包含许多现实世界视觉数据的对象，如CIFAR，FashionMNIST数据集等。
	每个TorchVision都包含两个参数：transform 和 target_transform 分别修改样本和标签。
	将 Dataset 作为一个参数传递给 Datadloader。这会对数据集进行循环，并支持 自动批处理、采样、洗牌和多进程数据加载。定义批次大小 batch_size= 某个数值，即每个元素 在 Dataloader 中，Iterable 会返回一批 batch_size 个特征和标签。

3、加载FashionMNIST数据集，包含以下参数：
training_data = datasets.FashionMNIST(
    root="data",                             #root是存储列车/测试数据的路径，
    train=True,                               #train指定训练或测试数据集，
    download=True,                       #download=True如果互联网上没有数据，就从互联网下载。root
    transform=ToTensor()               #transform并指定特征变换和标签变换target_transform
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

4、创建自定义数据集
     自定义数据集类必须实现三个函数：__init__、__len__和__getitem__
① 实例化 Dataset 对象时，__init__ 函数只运行一次。我们初始化 包含图片的目录、注释文件以及两个转换（已覆盖）
   如 def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
② __len__函数返回我们数据集中的样本数量  def __len__(self):
③ __getitem__函数加载并返回数据集中给定索引的样本。

5、遍历
① for batch in loader:  遍历全部 batch。
② 调试/特殊需求：next(iter(train_dataloader))，先 iter() 再 next()，想拿几批就 next 几批。

三、变换 torchvision.transforms
1、  ToTensor()   是 PyTorch 里最常见的 torchvision.transforms 之一。它把 PIL Image 或 numpy.ndarray 读取的图像 ⇒ 转换成 PyTorch 张量（torch.Tensor），并自动把 像素值从 [0,255] 缩放到 [0.0,1.0]，
	通道顺序：PIL/RGB ⇒ Tensor 会变成 C×H×W（通道在前）
	如果输入已经是 float32 且范围 0–1，ToTensor() 不会二次归一化。

2、Lambda变换应用任何用户自定义的lambda函数
 如  target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))     #把 0-9 的整数标签 y ⇒ 变成 10 维 one-hot 向量（float32）

四、构建神经网络
1、神经网络由层/模块组成，用于对数据进行作。
2、用 torch.nn 自己构建神经网络
    ①获取设备 
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")  #有加速器就用加速器，没有就用CPU
    ②构建方法:  定义类——》def __init__() 初始化神经网络层——》def forward定义数据流动方式，接收一批输入，返回这批数据的预测结果——》创建一个该类的实例model，并将其移到设备device，然后打印它的结构。
    ③关键点： 
	Model Layers，
	nn.Flatten，   将每个2D 28x28图像转换为连续的784像素数组
	nn.Linea     利用输入存储的权重和偏置对输入进行线性变换
	nn.ReLU，  在线性变换后应用以引入非线性，帮助神经网络学习各种各样的现象。
	nn.Sequential，是有序的 模块容器。数据按照定义的顺序通过所有模块传递。
	nn.Softmax，
	Model Parameters。

五、torch.autograd  自动求导。 PyTorch 内置的一个微分引擎，支持自动计算任意的梯度计算图。
      forward 就是模型的数据流水线，你把 tensor 扔进去，它把预测 tensor 吐出来，其余 PyTorch 全帮你搞定。
      loss.backward()                 # 自动回溯整个 forward 图
      forward 时 PyTorch 在后台动态建了计算图（DAG），每条边保存了“局部梯度函数”。   .backward() 只是从尾节点开始，沿这些边反向一路乘过去，直到所有叶子节点。

六、优化模型参数
① 超参数
	定义以下训练超参数：
	纪元数——在数据集上遍历的次数
	批处理大小——参数更新前通过网络传播的数据样本数量
	学习速率——每个批次/纪元更新模型参数的程度。
② 优化环路
	训练循环——遍历训练数据集，尝试收敛到最优参数。
	    optimizer.zero_grad()   #调用重置模型参数梯度
       	    loss.backward()     #反向传播预测损失
        	    optimizer.step()    #通过回传中收集的梯度来调整参数

	验证/测试循环——遍历测试数据集，检查模型性能是否在提升。
③ 损失函数
④ 优化器


数模型里到底有多少个可训练参数”的标配写法： total_params = sum(p.numel() for p in model.parameters())  遍历模型的每一块参数，用 .numel() 数每个张量里有多少个数，再全部加起来。
   model.parameters()  返回一个迭代器，里面依次给出模型中所有 nn.Parameter 张量（权重、偏置……）。
   tensor.numel() → “number of elements” 的缩写，返回该张量里总共有多少个数。

{变量:,}  或  {变量:_}    把整数（或浮点数）按每三位插入一个分隔符
   例如total_params = 12345678
         print(f"总参数:{total_params:,}")   # 总参数:12,345,678

plt.legend()  把当前坐标系里所有 带 label= 的曲线/散点/柱形自动收集起来，画一个图例框。如果之前画图时 没有写 label=，plt.legend() 会弹出空框或干脆不显示——先给元素起名字，再调 legend() 才有内容。

plt.savefig('linear_regression_compaison.png', dpi=150, bbox_inches='tight')
   dpi  每英寸像素数。150 足够网页/文档；打印可 300–600。
   bbox_inches='tight'  关键参数：自动裁剪多余空白区域，保证坐标轴、label、legend 都不被切掉；同时让图片尺寸“就内容而定”。
   一定要在 plt.show() 之前 调用 savefig，否则 show() 会清空图像。

except Exception as e:
        print(f"执行过程中出现错误:{e}")
        import traceback
        traceback.print_exc()
一旦出现异常，立即把完整的调用栈（哪个文件、哪一行、什么错误）原样打印到屏幕，方便你定位问题。