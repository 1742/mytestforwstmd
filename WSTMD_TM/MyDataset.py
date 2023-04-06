# coding=utf-8

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch


class MyDataset(Dataset):
    def __init__(self, data_dir, txt, ssw_txt, transform=None):
        """
        txt - 分类文件，包含各文件类别
        ssw_txt - 应该是对应文件的齿痕位置。。。
        """
        super(MyDataset, self).__init__()
        self.ssw_txt = open(ssw_txt, 'r')
        data = []
        # ???还不知道干啥的
        minlen=1000
        fh = open(txt, 'r')
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表，按行循环txt文本中的内
            # 删除 本行string 字符串开头和末尾的指定字符，这个方法的详细介绍自己查询python
            line = line.strip('\n')
            line = line.rstrip('\n')
            # words = [文件名，类别]
            words = line.split()
            for linee in self.ssw_txt:  # 迭代该列表#按行循环txt文本中的内
                linee = linee.rstrip()
                wordss = linee.split()
                # 没有的对分类文件和检测文件的顺序对齐，需要一个一个比对匹配相同文件的类别和检测框。。。。。。。。
                if wordss[0] == words[0]:
                    # ssw_block记录检测框数量（行数）和每个检测框的坐标及大小（4列）
                    ssw_block = torch.Tensor(int((len(wordss) - 1) / 4), 4)
                    for i in range(int((len(wordss) - 1) / 4)):
                        # ssw_block[i, 0] = int(wordss[i * 4 + 1])
                        # ssw_block[i, 1] = int(wordss[i * 4 + 2])
                        # ssw_block[i, 2] = int(wordss[i * 4 + 3])
                        # ssw_block[i, 3] = int(wordss[i * 4 + 4])
                        # ssw中每4个数字组成一个检测框，[横坐标,纵坐标,长,宽]
                        # 除以32是计算出该坐标在深度特征图上的位置
                        ssw_block[i, 0] = round(float(wordss[i * 4 + 1]) / 32)
                        ssw_block[i, 1] = round(float(wordss[i * 4 + 2]) / 32)
                        ssw_block[i, 2] = round(float(wordss[i * 4 + 3]) / 32)
                        ssw_block[i, 3] = round(float(wordss[i * 4 + 4]) / 32)
                    if minlen > len(ssw_block):
                        minlen = len(ssw_block)
                    # 找到对应的类别和检测框后结束该文件的匹配搜索
                    break
            data.append((words[0], ssw_block, int(words[1])))

        self.data=data
        self.minlen=minlen
        self.data_dir = data_dir
        self.transform = transform


    def __len__(self):
        return len(self.data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        fn, ssw, label = self.data[index]
        image_path = os.path.join(self.data_dir, fn)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, ssw[:self.minlen], label
        #
        # return image, ssw, label


if __name__ == '__main__':
    img_path = r'data/img'
    train_txt_path = r'txt/train1_t.txt'
    ssw_path = 'ssw_5.txt'

    dataset = MyDataset(img_path, train_txt_path, ssw_path)
    dataset.__getitem__(24)
