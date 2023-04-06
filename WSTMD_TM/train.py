import os
import sys
import time

from model_wsddn import WSDDN_res
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import warnings
from MyDataset import MyDataset

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 100
early_stop_step = 20
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224

model_use_pretrain_weight = True
image_path = 'data/img'  # tongue data set path
train_txt_path = 'txt/train1_t.txt'
val_txt_path = 'txt/val1_t.txt'
ssw_path = 'ssw_5.txt'
save_name = 'Resnet34_WSDDN'

# print("device is " + str(torch.cuda.get_device_name()))

def get_confusion_matrix(trues, preds):
    labels = [0, 1]
    conf_matrix = confusion_matrix(trues, preds)
    return conf_matrix


def roc_auc(trues, preds):
    fpr, tpr, thresholds = roc_curve(trues, preds)
    auc = roc_auc_score(trues, preds)
    return fpr, tpr, auc


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_confusion_matrix(conf_matrix):
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1]
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


def main(EPOCH, model_use_pretrain_weight, image_path, train_txt_path, val_txt_path,
         ssw_path, save_name):

    best_accuracy = 0.0
    trigger = 0

    val_loss = []
    val_presision = []
    val_recall = []
    val_f1 = []
    val_accuracy = []
    val_conf_matrix = []
    model = WSDDN_res()

    if model_use_pretrain_weight:
        # model_weight_path = "resnet34-333f7ec4.pth"
        model_weight_path = "pre_weight/resnet34_t_pre_1.pkl"
        model_weight_path_web = r'C:\Users\13632\.cache\torch\hub\pytorch_vision_v0.10.0'
        # model_weight_path = "resnet34_test.pkl"
        # model_weight_path = "vgg16-397923af.pth"
        # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        if not os.path.exists(model_weight_path):
            if not os.path.exists(model_weight_path_web):
                print('Select pre-trained model weights. But not found the file. Download from website.')
                # 这句用来防止下载出错
                torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            else:
                print('Loading weight from {}'.format(model_weight_path_web))
            pre_weights = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
            pre_weights = pre_weights.state_dict()
        else:
            pre_weights = torch.load(model_weight_path, map_location=device)
            print('Successfully load weights from local files')
            print(pre_weights)

        # 删除resnet34最后两层fc
        del_key = []
        for key, _ in pre_weights.items():
            if "fc" in key:
                del_key.append(key)
        for key in del_key:
            del pre_weights[key]

        missing_keys, unexpected_keys = model.load_state_dict(pre_weights, strict=False)
        print("[missing_keys]:", *missing_keys, sep="\n")
        print("[unexpected_keys]:", *unexpected_keys, sep="\n")
        print("\n")

    print(model)
    print('params:' + str(count_params(model)) + '\n')

    model.to(device)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),

        "val": transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    train_dataset = MyDataset(data_dir=image_path, txt=train_txt_path, ssw_txt=ssw_path,
                              transform=data_transform['train'])
    val_dataset = MyDataset(data_dir=image_path, txt=val_txt_path, ssw_txt=ssw_path, transform=data_transform['val'])

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0)

    data_loader_val = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    # torch.optim.lr_scheduler - 学习率衰减
    # ExponentialLR - 新学习率 = 学习率 * gamma的epoch次方
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    print("start training")

    before = time.time()
    for epoch in range(EPOCH):
        tot_train_loss = 0.0
        tot_val_loss = 0.0

        train_preds = []
        train_trues = []

        model.train()
        with tqdm(total=len(data_loader_train)) as pbar:
            for i, (train_data_batch, train_box_batch, train_label_batch) in enumerate(data_loader_train):
                pbar.set_description('epoch - {}'.format(epoch))
                train_data_batch = train_data_batch.float().to(device)  # 将double数据转换为float
                train_label_batch = train_label_batch.to(device)
                train_box_batch = train_box_batch.float().to(device)

                train_outputs, train_op2, train_op3 = model(train_data_batch, train_box_batch)
                loss = criterion(train_outputs, train_label_batch)
                # print(loss)
                # 反向传播优化网络参数
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 累加每个step的损失
                tot_train_loss += loss.data
                train_outputs = train_outputs.argmax(dim=1)

                train_preds.extend(train_outputs.detach().cpu().numpy())
                train_trues.extend(train_label_batch.detach().cpu().numpy())

            train_accuracy = accuracy_score(train_trues, train_preds)
            train_precision = precision_score(train_trues, train_preds)
            train_recall = recall_score(train_trues, train_preds)
            train_f1 = f1_score(train_trues, train_preds)

            # print("[train] Epoch:{} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} loss:{:.4f}".format(
            #     epoch, train_accuracy, train_precision, train_recall, train_f1, tot_train_loss))

            val_preds = []
            val_trues = []

            model.eval()
            # 在with torch.no_grad()下所有tensor的required_grad设置为False，提高运算速度
            # 验证集不需要反向传播
            with torch.no_grad():
                for i, (val_data_batch, val_box_batch, val_label_batch) in tqdm(enumerate(data_loader_val),
                                                                                total=len(data_loader_val)):
                    val_data_batch = val_data_batch.float().to(device)  # 将double数据转换为float
                    val_label_batch = val_label_batch.to(device)
                    val_box_batch = val_box_batch.float().to(device)
                    val_outputs, val_op2, val_op3 = model(val_data_batch, val_box_batch)

                    loss = criterion(val_outputs, val_label_batch)
                    tot_val_loss += loss.data
                    val_outputs = val_outputs.argmax(dim=1)

                    val_preds.extend(val_outputs.detach().cpu().numpy())
                    val_trues.extend(val_label_batch.detach().cpu().numpy())

                val_accuracy = accuracy_score(val_trues, val_preds)
                val_precision = precision_score(val_trues, val_preds)
                val_recall = recall_score(val_trues, val_preds)
                val_f1 = f1_score(val_trues, val_preds)
                conf_matrix = get_confusion_matrix(val_trues, val_preds)

                # print("[val] Epoch:{} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} loss:{:.4f} ".format(epoch,
                #                                                                                                     accuracy,
                #                                                                                                     precision,
                #                                                                                                     recall,
                #                                                                                                     f1,
                #                                                                                                     tot_val_loss))

                trigger += 1
                if val_accuracy >= best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(model.state_dict(), "./" + save_name + '.pkl')
                    print("save best weighted ")
                    print("best_accuracy:{:.4f}".format(best_accuracy))
                    # trigger = 0
                    #
                    # if train_accuracy>train_best_accuracy :
                    #     train_best_accuracy=train_accuracy
                    trigger = 0

                if trigger >= early_stop_step:
                    print("=> early stopping")
                    break

                # print(classification_report(val_trues, val_preds))
                # print(conf_matrix)+
                # if epoch == EPOCH - 1:
                #     plot_confusion_matrix(conf_matrix)

                val_accuracy.append(val_accuracy), val_presision.append(val_precision), val_recall.append(val_recall), val_f1.append(
                    val_f1), val_loss.append(tot_val_loss.item()), val_conf_matrix.append(conf_matrix)

            pbar.set_postfix({'trian_acc':train_accuracy, 'train_precision':train_precision, 'train_recall':train_recall, 'train_f1':train_f1, 'train_loss':tot_train_loss,
                              'val_acc':val_accuracy, 'val_precision':val_precision, 'val_recall':val_recall, 'val_f1':val_f1, 'val_loss':tot_val_loss})
            pbar.update(1)

    result_path = 'result_' + save_name
    np.savez(result_path, val_accuracy=val_accuracy, val_presision=val_presision, val_recall=val_recall, val_f1=val_f1,
             val_loss=val_loss, val_conf_matrix=val_conf_matrix)
    after = time.time()
    total_time = after - before
    print('total_time: ' + str(total_time / 60) + ' min')
    print('best_accuracy: ' + str(best_accuracy))
    print('trigger: ' + str(trigger))


if __name__ == '__main__':
    image_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\WSTMD\WSTMD-main\WSTMD_TM\mydata\img'
    train_txt_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\WSTMD\WSTMD-main\WSTMD_TM\mydata\my_train_data.txt'
    val_txt_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\WSTMD\WSTMD-main\WSTMD_TM\mydata\my_val_data.txt'
    test_txt_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\WSTMD\WSTMD-main\WSTMD_TM\mydata\my_test_data.txt'
    ssw_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\WSTMD\WSTMD-main\WSTMD_TM\mydata\myssw.txt'

    # from data_pre2 import box11
    # myssw = []
    # for path in [train_txt_path, val_txt_path, test_txt_path]:
    #     with open(path, 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             img_name = line.split(' ')[0]
    #             img_path = os.path.join(image_path, img_name + '.jpg')
    #             myssw.append([img_name, box11(img_path)])
    #
    # with open(ssw_path, 'w', encoding='utf-8') as f:
    #     for ssw in myssw:
    #         for i in ssw:
    #             if type(i) == type('a'):
    #                 f.write(i + '.jpg')
    #                 f.write(' ')
    #             else:
    #                 for j in i:
    #                     f.write(str(j) + ' ')
    #         f.write('\n')
    #
    # print('Successfully generate candidate region txt!')

    main(EPOCH, model_use_pretrain_weight, image_path, train_txt_path, val_txt_path,
         ssw_path, save_name)
