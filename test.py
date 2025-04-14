from dataloader.dataloader import XJTUdata,MITdata,HUSTdata,TJUdata, HUST_test_dataloader
from utils.metric import Test
from Model.Model import PINN
import argparse
import os
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for HUST dataset')
    parser.add_argument('--data', type=str, default='HUST', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--data_path', type=str, default='../../bishe_data/data/HUST_data', help='battery data path')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization', type=bool, default=True, help='normalization processing or not')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')

    # model related
    # parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    # parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default=None, help='save folder')
    parser.add_argument('--load_path', type=str, default='Model/ckpt/best_model_0412.pth', help='checkpoint path')

    args = parser.parse_args()

    return args

def load_HUST_data(args,small_sample=None):
    test_id = ['1-4','1-8','2-4','2-8',
               '3-4','3-8','4-4','4-8',
               '5-4','5-7','6-4','6-8',
               '7-4','7-8','8-4','8-8',
               '9-4','9-8','10-4','10-8']
    data = HUSTdata(root='../bishe_data/data/HUST_data',args=args)
    train_list = []
    test_list = []
    files = os.listdir('../bishe_data/data/HUST_data')
    for f in files:
        if f[:-4] in test_id:
            test_list.append(f'../bishe_data/data/HUST_data/{f}')
        else:
            train_list.append(f'../bishe_data/data/HUST_data/{f}')
    if small_sample is not None:
        train_list = train_list[:small_sample]

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def main():
    test_id = ['1-4','1-8','2-4','2-8',
            '3-4','3-8','4-4','4-8',
            '5-4','5-7','6-4','6-8',
            '7-4','7-8','8-4','8-8',
            '9-4','9-8','10-4','10-8']
    args = get_args()

    timestamp = datetime.now().strftime('%m_%d_%H%M%S')
    setattr(args, 'load_path', f'Model/ckpt/best_model_0412.pth')
    pinn = PINN(args)
    pinn.load_model(args.load_path)

    # dataloader = load_HUST_data(args)
    for i in range(len(test_id)):
        test_data = test_id[i]+'.csv'
        test_path = os.path.join(args.data_path,test_data)
        test_dataloader = HUST_test_dataloader(test_path,nominal_capacity=1.19, nominal_rul=3000)
        label,pred = Test(pinn, test_dataloader)



    
if __name__ == '__main__':
    main()