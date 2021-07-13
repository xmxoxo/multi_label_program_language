# 多标签模型：程序语言预测

2021/3/2


数据集：program language
数据中是一个英文文本，标签是文本对应的程序语言标签；
练习构造多标签模型对文本进行训练与预测。

## 数据预处理

数据集情况: 英文
训练集：train.tsv  10万条
验证集：validation.tsv  3万条
测试集：test.tsv 2万条

字段名：title,tags; 测试集没有tags字段

先把所有的tags合并，分析下大小：
总共有100种，最大的tags大小为5；

模拟训练与预测：
```
python task_program_language.py --epochs=3 --sim=1
python task_program_language.py --task=predict --sim=1
```

模拟过程：

```
18:16:39.58|F:>python task_program_language.py --epochs=1 --sim=1
2021-03-02 18:16:50.815303: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic libra
ry 'cudart64_100.dll'; dlerror: cudart64_100.dll not found
2021-03-02 18:16:50.826303: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do n
ot have a GPU set up on your machine.
Using TensorFlow backend.
运行参数: 模拟=1, task=train, models=./models/, epochs=1, batch_size=32, learning_rate=0.000020
训练数据预处理...
训练数据已存在,跳过处理过程。
<class 'list'>
32
[['Random Battleship Placement Method', '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'], ['React.js, wai
t for setState to finish before triggering a function?', '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]']
, ['How to match nearest latitude and longitude in Array using PHP?', '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0]'], ["PHP: get contents of li's within ul", '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'],
['Cannot create TypedQuery for query with more than one return using requested result type', '[0, 0, 0, 0, 0, 0, 1, 0, 0
, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]']]
正在训练模型...
WARNING:tensorflow:From C:\ProgramData\Miniconda3\lib\site-packages\keras\backend\tensorflow_backend.py:422: The name tf
.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/1
1/1 [==============================] - 14s 14s/step - loss: 0.7366 - accuracy: 0.4884 - val_loss: 0.6990 - val_accuracy:
 0.5250

Epoch 00001: val_accuracy improved from -inf to 0.52500, saving model to ./models/best.h5
正在保存训练曲线数据...
训练曲线数据保存完成...
模型权重已加载.
<class 'list'>
32
self.test_data: [['Triangular array', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'], ['java swing - JL
abel not rotating', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'], ['What is Twig for php template eng
ine?', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'], ['Python multithreading crawler', '[0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'], ['C# - Waiting for a copy operation to complete', '[0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]']]
1/1 [==============================] - 4s 4s/step
预测结果已保存。

18:17:53.20|F:>
```

Loss函数： binary_crossentropy

在服务器上训练与预测：
```
python task_program_language.py --epochs=3
python task_program_language.py --task=predict
```

训练过程：

```
Epoch 1/20
2021-03-02 18:03:43.055331: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
3125/3125 [==============================] - 499s 160ms/step - loss: 0.0551 - accuracy: 0.9867 - val_loss: 0.0320 - val_accuracy: 0.9900

Epoch 00001: val_accuracy improved from -inf to 0.98996, saving model to ./models/best.h5
Epoch 2/20
3125/3125 [==============================] - 491s 157ms/step - loss: 0.0324 - accuracy: 0.9903 - val_loss: 0.0397 - val_accuracy: 0.9903

Epoch 00002: val_accuracy improved from 0.98996 to 0.99026, saving model to ./models/best.h5
```
模型改名为：`models/best01.h5`
预测结果保存为：`predict1.tsv`

**Loss函数对比**

换成另一个Loss函数：`categorical_crossentropy` 做下对比试一下：
使用这个Loss收敛比较慢，要多几轮才行：

```
python task_program_language.py --models=./models01/ --epochs=10
```

训练过程：

```
root@ubuntu:/mnt/sda1/transdat/complex_models/multi_label_program_language# python task_program_language.py --models=./models01/ --epochs=10
Using TensorFlow backend.
运行参数: 模拟=0, task=train, models=./models01/, epochs=10, batch_size=32, learning_rate=0.000020
训练数据预处理...
训练数据已存在,跳过处理过程。
.....
正在训练模型...
.....
Epoch 1/10
2021-03-02 19:04:24.859095: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
3125/3125 [==============================] - 499s 160ms/step - loss: 4.5970 - accuracy: 0.6090 - val_loss: 3.2789 - val_accuracy: 0.6439

Epoch 00001: val_accuracy improved from -inf to 0.64390, saving model to ./models01/best.h5
Epoch 2/10
3125/3125 [==============================] - 491s 157ms/step - loss: 3.4567 - accuracy: 0.6705 - val_loss: 2.5956 - val_accuracy: 0.6495

Epoch 00002: val_accuracy improved from 0.64390 to 0.64950, saving model to ./models01/best.h5
Epoch 3/10
3125/3125 [==============================] - 492s 157ms/step - loss: 3.1652 - accuracy: 0.6870 - val_loss: 2.6170 - val_accuracy: 0.6440

Epoch 00003: val_accuracy did not improve from 0.64950
Epoch 4/10
3125/3125 [==============================] - 491s 157ms/step - loss: 2.9329 - accuracy: 0.7036 - val_loss: 4.4642 - val_accuracy: 0.6563

Epoch 00004: val_accuracy improved from 0.64950 to 0.65627, saving model to ./models01/best.h5
Epoch 5/10
3125/3125 [==============================] - 492s 157ms/step - loss: 2.7195 - accuracy: 0.7156 - val_loss: 2.9422 - val_accuracy: 0.6471

Epoch 00005: val_accuracy did not improve from 0.65627
Epoch 6/10
3125/3125 [==============================] - 491s 157ms/step - loss: 2.5282 - accuracy: 0.7263 - val_loss: 2.8224 - val_accuracy: 0.6361

Epoch 00006: val_accuracy did not improve from 0.65627
Epoch 7/10
3125/3125 [==============================] - 491s 157ms/step - loss: 2.3607 - accuracy: 0.7251 - val_loss: 5.0127 - val_accuracy: 0.6283

Epoch 00007: val_accuracy did not improve from 0.65627
Epoch 8/10
3125/3125 [==============================] - 491s 157ms/step - loss: 2.2143 - accuracy: 0.7207 - val_loss: 2.6220 - val_accuracy: 0.6201

Epoch 00008: val_accuracy did not improve from 0.65627
Epoch 9/10
3125/3125 [==============================] - 491s 157ms/step - loss: 2.0900 - accuracy: 0.7173 - val_loss: 3.4246 - val_accuracy: 0.6190

Epoch 00009: val_accuracy did not improve from 0.65627
Epoch 10/10
3125/3125 [==============================] - 491s 157ms/step - loss: 1.9883 - accuracy: 0.7122 - val_loss: 2.6866 - val_accuracy: 0.5998

Epoch 00010: val_accuracy did not improve from 0.65627
正在保存训练曲线数据...
训练曲线数据保存完成...
模型权重已加载.
625/625 [==============================] - 22s 35ms/step
预测结果已保存。
```


参考资料：

使用python和sklearn的文本多标签分类实战开发
https://blog.csdn.net/weixin_42608414/article/details/88100879

关于损失函数和评估标准：
https://keras.io/zh/losses/
https://keras.io/zh/metrics/

https://baijiahao.baidu.com/s?id=1670661688802228352&wfr=spider&for=pc

多标签模型评价方法
https://blog.csdn.net/weixin_37801695/article/details/86496754


多分类多标签模型的评估方式（定义+numpy代码实现）
https://blog.csdn.net/Nin7a/article/details/113503781


多标签分类的评价指标_hzhj的博客-CSDN博客_多标签分类评价指标
https://blog.csdn.net/hzhj2007/article/details/79153647



评估模型：

```
python task_program_language.py --task=eval --models=./models01/ --sim=1

1/1 [==============================] - 4s 4s/step
accuracy: 0.40625
f1_score_macro: 0.18820969089390138
f1_score_micro: 0.7130434782608696
f1_score_weighted: 0.668322688441534
```

在GPU上进行完整的评估：
版本依赖: scikit-learn==0.22.2
每个标签AUC结果保存在：`models01/measure_auc.tsv`

```
python task_program_language.py --task=eval --models=./models01/

938/938 [==============================] - 30s 32ms/step
accuracy: 0.41073333333333334
f1_score_macro: 0.5775857500010448
f1_score_micro: 0.7177718108587813
f1_score_weighted: 0.6993239212930686

val set mean column auc:    label_name       auc
0           r  0.879856
1         php  0.910727
2       mysql  0.807572
3          c#  0.891275
4  javascript  0.919705

```
#-----------------------------------------

使用 Loss函数:`binary_crossentropy` 重新训练模型：

```
python task_program_language.py --epochs=3

Using TensorFlow backend.
运行参数: 模拟=0, task=train, models=./models/, epochs=3, batch_size=32, learning_rate=0.000020
训练数据预处理...
训练数据已存在,跳过处理过程。
正在训练模型...

Epoch 1/3
2021-03-02 23:07:02.123940: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
3125/3125 [==============================] - 497s 159ms/step - loss: 0.0548 - accuracy: 0.9866 - val_loss: 0.0294 - val_accuracy: 0.9900

Epoch 00001: val_accuracy improved from -inf to 0.99003, saving model to ./models/best.h5
Epoch 2/3
3125/3125 [==============================] - 492s 157ms/step - loss: 0.0323 - accuracy: 0.9903 - val_loss: 0.0312 - val_accuracy: 0.9902

Epoch 00002: val_accuracy improved from 0.99003 to 0.99023, saving model to ./models/best.h5
Epoch 3/3
3125/3125 [==============================] - 491s 157ms/step - loss: 0.0283 - accuracy: 0.9911 - val_loss: 0.0300 - val_accuracy: 0.9904

Epoch 00003: val_accuracy improved from 0.99023 to 0.99040, saving model to ./models/best.h5
正在保存训练曲线数据...
训练曲线数据保存完成...
模型权重已加载.
625/625 [==============================] - 22s 35ms/step
预测结果已保存。
正在评估模型...
938/938 [==============================] - 31s 33ms/step
accuracy: 0.42013333333333336
f1_score_macro: 0.5664253272695015
f1_score_micro: 0.7245645232222414
f1_score_weighted: 0.7021235984467628
hamming_loss: 0.009603333333333297(0.9903966666666667)

val set mean column auc:    label_name       auc
0           r  0.888365
1         php  0.904091
2       mysql  0.819662
3          c#  0.879840
4  javascript  0.912893

```



