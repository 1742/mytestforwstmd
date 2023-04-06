from train import *

if __name__ == '__main__':
    image_path = r'content\drive\MyDrive\Colab Notebooks\WSTMD\mydata\img'
    train_txt_path = r'content\drive\MyDrive\Colab Notebooks\WSTMD\mydata\my_train_data.txt'
    val_txt_path = r'content\drive\MyDrive\Colab Notebooks\WSTMD\mydata\my_val_data.txt'
    test_txt_path = r'content\drive\MyDrive\Colab Notebooks\WSTMD\mydata\my_test_data.txt'
    ssw_path = r'content\drive\MyDrive\Colab Notebooks\WSTMD\mydata\myssw.txt'

    main(EPOCH, model_use_pretrain_weight, image_path, train_txt_path, val_txt_path,
         ssw_path, save_name)
