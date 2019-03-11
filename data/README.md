データセットは以下のように読み込むことができる．
読み込まれたデータは numpy の ndarray になっている。


```python
import numpy as np

train_image = np.load('/path/to/train_image.npy')
train_label = np.load('/path/to/train_label.npy')
```
