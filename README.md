cnet
====

cnet is a neural network for C programmers.

开始
----

使用cnet很简单。

首先，定义网络结构，以一个12层的网络为例
``` c
	net_t *n = net_create(12);
	layer_t *dropout = dropout_layer(0, 0);
```
前三层分别是

- 卷积 输入 1 x 28 x 28 输出 32 x 28 x 28 卷积核 5 x 5 步长 1 无填充
- 激活函数 Relu
- 最大池化 输入 32 x 28 x 28 输出 32 x 14 x 14
``` c
	net_add(n, conv_layer(1, 28, 28, 32, 28, 28, 5, 1, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, max_pooling_layer(32, 28, 28, 14, 14, 2, 0, 0));
```
可以注意到，有些参数直接设置为0了，但这并不会导致错误。cnet会根据已知信息，自动调整设置为0的参数。
在上面的例子中，卷积的填充参数，relu的输入、输出，池化的步长、填充参数都是这种情况。

接下来是类似的三层
``` c
	net_add(n, conv_layer(32, 14, 14, 64, 14, 14, 5, 1, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, max_pooling_layer(64, 14, 14, 7, 7, 2, 0, 0));
```
现在输出为 64 x 7 x 7。接下来是全连接层
``` c
	net_add(n, fc_layer(0, 1024, 0));
	net_add(n, relu_layer(0, 0, 0));
	net_add(n, dropout);
```
这里同样省略了大量的参数，我们只需要告诉cnet最终的输出是1024维的向量。
另一点是这里使用了前面创建的dropout层，dropout层的指针要留出来后面要用。

最后是输出层
``` c
	net_add(n, fc_layer(0, 10, 0));
	net_add(n, softmax_layer(0, 0, 0));
	net_add(n, cee_layer(0, 0, 0));
```
这里使用softmax做了一个10类的输出，并在其后接了一个交叉熵损失函数。
网络的最后一层通常需要是某种损失函数，训练时，会最小化这个损失函数。

现在可以告诉cnet我们的网络结构定义完成了。
``` c
    net_finish(n, TRAIN_ADAM);
```
`TRAIN_ADAM`参数的意思是告诉cnet我们要使用ADAM算法对网络进行训练。
训练的过程也很简单，通常是写在一个循环中，不过，再开始之前也许你会考虑加载一些预训练的参数。
``` c
    net_param_load(n, "params.bin");

    SET_DROP_PROB(dropout, 0.5);
    for (i = 0; i < 1000; ++i)
	{
		net_train(n, feed_data, rate, images->dim[0] / 100);
    }

    net_param_save(n, "params.bin");
```
在上面的代码中，我展示了如何加载和保存训练参数。
训练只需调用`net_train`函数即可，`feed_data`是你需要实现的向网络提供数据的回调函数。
`rate`是学习率，但在ADAM的情况下，这个参数将被忽略。
最后一个参数指出了一个批次的训练数据的量，它通常被称为`batch_size`。

注意在训练开始前我使用`SET_DROP_PROB`宏将dropout层的丢弃率设为了0.5。
这会使得训练不至于过拟合，但在验证阶段，你应该将dropout层关闭，即设置丢弃率为0。
``` c
	SET_DROP_PROB(dropout, 0);
	for (i = 0; i < images->dim[0]; ++i)
	{
		feed_data(n);
		net_forward(n);

		right += (arg_max(LAST_LAYER(n)->in.val, 10) == arg_max(LAST_LAYER(n)->param.val, 10));
	}

	LOG("accurcy %f\n", 1.0 * right / images->dim[0]);
```
`LAST_LAYER`可以取得最后一层的指针，通过比较输入和参数，我们可以得到网络预测正确的数目。
需要说明的是这里`feed_data`已经由训练数据集切换到了验证数据集。详细的代码请参考example文件夹中的[mnist_example.c](example/mnist_example.c)。

祝你玩得开心。

后续计划
-------

- GPU训练支持
- 更多种类的层
- ...

当前版本的性能信息请参考[wiki](https://github.com/yang-le/cnet/wiki)

如果你有好的想法、意见、建议，欢迎提交[issue](https://github.com/yang-le/cnet/issues)。

如果你实现了新的层或者训练算法，欢迎分享，Fork并发起[Pull Request](https://github.com/yang-le/cnet/pulls)。

Happy Hacking!
