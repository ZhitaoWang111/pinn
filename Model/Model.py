import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from utils.util import AverageMeter,get_logger,eval_metrix
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self,input_dim=17,output_dim=1,layers_num=4,hidden_dim=50,droupout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        x = self.net(x)
        return x


class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,2)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)

class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        self.encoder = MLP(input_dim=17,output_dim=32,layers_num=3,hidden_dim=60,droupout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)


def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr



class PINN(nn.Module):
    def __init__(self,args):
        super(PINN, self).__init__()
        self.args = args
        board_dir = None
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
            board_dir = os.path.join(args.save_folder, 'run')
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()
        if board_dir is not None and not os.path.exists(board_dir):
            os.makedirs(board_dir)
        self.writer = SummaryWriter(log_dir=board_dir)

        self.solution_u = Solution_u().to(device)
        self.dynamical_F = MLP(input_dim=53,output_dim=2,
                               layers_num=args.F_layers_num,
                               hidden_dim=args.F_hidden_dim,
                               droupout=0.2).to(device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.warmup_lr)
        self.optimizer1 = torch.optim.Adam(self.solution_u.parameters(), lr=args.warmup_lr)
        self.optimizer2 = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr_F)

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # 模型的最好参数(the best model)
        self.best_model = None

        # loss = loss1 + alpha*loss2 + beta*loss3
        self.alpha = self.args.alpha
        self.beta = self.args.beta

    def _save_args(self):
        if self.args.log_dir is not None:
            # 把parser中的参数保存在self.logger中
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        for param in self.solution_u.parameters():
            param.requires_grad = True

    def predict(self,xt):
        return self.solution_u(xt)

    def Test(self,testloader):
        self.eval()
        label = []
        pred = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(testloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                label.append(y1)
                pred.append(u1.cpu().detach().numpy())
        pred = np.concatenate(pred,axis=0)
        label = np.concatenate(label,axis=0)

        return label,pred

    def Valid(self,validloader):
        self.eval()
        label = []
        pred = []
        with torch.no_grad():
            for iter,(x1,_,y1,_) in enumerate(validloader):
                x1 = x1.to(device)
                u1 = self.predict(x1)
                label.append(y1)
                pred.append(u1.cpu().detach().numpy())
        pred = np.concatenate(pred,axis=0)
        label = np.concatenate(label,axis=0)
        pred_soh = pred[:,0]
        pred_rul = pred[:,1]
        label_soh = label[:,0]
        label_rul = label[:,1]
        mse_soh = self.loss_func(torch.tensor(pred_soh),torch.tensor(label_soh))
        mse_rul = self.loss_func(torch.tensor(pred_rul),torch.tensor(label_rul))
        mse = mse_soh + mse_rul
        return mse.item(), mse_soh.item(), mse_rul.item()

    def forward(self,xt):
        xt.requires_grad = True
        x = xt[:,0:-1]
        t = xt[:,-1:]

        u = self.solution_u(torch.cat((x,t),dim=1))     # 对应论文中的 F 函数
        pred_soh = u[:, 0:1]
        pred_rul = u[:, 1:2]

        pred_soh_t = grad(pred_soh.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        pred_rul_t = grad(pred_rul.sum(),t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        pred_soh_x = grad(pred_soh.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        pred_rul_x = grad(pred_rul.sum(),x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        
        u_t = torch.cat([pred_soh_t, pred_rul_t], dim=1)
        u_x = torch.cat([pred_soh_x, pred_rul_x], dim=1)

        F = self.dynamical_F(torch.cat([xt,u,u_x,u_t],dim=1))   # 对应论文中的 G 函数； G 函数应该近似于 du/dt 

        f = u_t - F
        return u,f

    def train_one_epoch(self,epoch,dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss1_soh_meter = AverageMeter()
        loss1_rul_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # nn.MSELoss按照对应位置计算差值的平方后，对(batch_size，2)个数据取平均值
            # data loss
            loss1_soh = 0.5*self.loss_func(u1[:,0],y1[:,0]) + 0.5*self.loss_func(u2[:,0],y2[:,0])
            loss1_rul = 0.5*self.loss_func(u1[:,1],y1[:,1]) + 0.5*self.loss_func(u2[:,1],y2[:,1])
            # loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)
            loss1 = loss1_soh*0.3+loss1_rul*0.7

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss1_meter.update(loss1.item())
            loss1_soh_meter.update(loss1_soh.item())
            loss1_rul_meter.update(loss1_rul.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg, loss1_soh_meter.avg, loss1_rul_meter.avg, loss2_meter.avg, loss3_meter.avg

    def Train(self,trainloader,testloader=None,validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        min_RMSE_soh = 5
        min_RMSE_rul = 3000
        columns = ['loss', 'loss1', 'loss1_soh', 'loss1_rul', 'loss_2', 'loss_3', 'valid_mse','valid_mse_soh','valid_mse_rul']
        df = pd.DataFrame(columns=columns)
        for e in range(1,self.args.epochs+1):
            early_stop += 1
            # 训练及相关参数保存
            loss1,loss1_soh,loss1_rul,loss2,loss3 = self.train_one_epoch(e,trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e,current_lr,loss1+self.alpha*loss2+self.beta*loss3)
            self.logger.info(info)
            df.loc[e, 'loss'] = loss1+self.alpha*loss2+self.beta*loss3
            df.loc[e, 'loss1'] = loss1
            df.loc[e, 'loss1_soh'] = loss1_soh
            df.loc[e, 'loss1_rul'] = loss1_rul
            df.loc[e, 'loss_2'] = loss2
            df.loc[e, 'loss_3'] = loss3
            self.writer.add_scalar('Total_loss', loss1 + self.alpha * loss2 + self.beta * loss3, e)
            self.writer.add_scalar('Loss1', loss1, e)
            self.writer.add_scalar('Loss1_soh', loss1, e)
            self.writer.add_scalar('Loss1_rul', loss1, e)
            self.writer.add_scalar('Loss2', loss2, e)
            self.writer.add_scalar('Loss3', loss3, e)
            # 验证
            if e % 1 == 0 and validloader is not None:
                valid_mse, valid_mse_soh, valid_mse_rul = self.Valid(validloader)
                info = '-----[Valid]-----\n' \
                ' epoch:{}, MSE: {}\n' \
                ' MSE_soh: {}, MSE_rul: {}'.format(e,valid_mse, valid_mse_soh, valid_mse_rul)
                self.logger.info(info)
                df.loc[e, 'valid_mse'] = valid_mse
                df.loc[e, 'valid_mse_soh'] = valid_mse_soh
                df.loc[e, 'valid_mse_rul'] = valid_mse_rul
                self.writer.add_scalar('valid_mse', valid_mse, e)
                self.writer.add_scalar('valid_mse_soh', valid_mse_soh, e)
                self.writer.add_scalar('valid_mse_rul', valid_mse_rul, e)

            # 测试 对于best_model进行test，并记录test指标
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                label,pred = self.Test(testloader)
                pred_soh = pred[:,0]*1.19
                pred_rul = pred[:,1]*3000
                label_soh = label[:,0]*1.19
                label_rul = label[:,1]*3000
                [MAE_soh, MAPE_soh, MSE_soh, RMSE_soh] = eval_metrix(pred_soh, label_soh)
                [MAE_rul, MAPE_rul, MSE_rul, RMSE_rul] = eval_metrix(pred_rul, label_rul)
                if RMSE_soh < min_RMSE_soh:
                    min_soh_e = e
                    min_RMSE_soh = RMSE_soh
                if RMSE_rul < min_RMSE_rul:
                    min_rul_e = e
                    min_RMSE_rul = RMSE_rul
                info = '-----[Test]-----\n' \
                ' MSE_soh: {:.8f}, MAE_soh: {:.6f}, MAPE_soh: {:.6f}, RMSE_soh: {:.6f}\n' \
                ' MSE_rul: {:.8f}, MAE_rul: {:.6f}, MAPE_rul: {:.6f}, RMSE_rul: {:.6f}'.format(MSE_soh, MAE_soh, MAPE_soh, RMSE_soh,MSE_rul, MAE_rul, MAPE_rul, RMSE_rul)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u':self.solution_u.state_dict(),
                                   'dynamical_F':self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'label.npy'), label)
                    np.save(os.path.join(self.args.save_folder, 'pred.npy'), pred)
                ##################################################################################
            # 如果多次训练MSE都不再降低，则将程序early_stop
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        self.writer.close()
        if self.args.save_folder is not None:
            df.to_csv(os.path.join(self.args.save_folder,'loss_mse.csv'), index=False, float_format='%.6f')
            torch.save(self.best_model,os.path.join(self.args.save_folder,'best_model.pth'))
        print(f'训练结束\n test_RMSE_soh最小的循环来自epoch{min_soh_e}，为{min_RMSE_soh}, \n \
              test_RMSE_rul最小的循环来自epoch{min_rul_e}，为{min_RMSE_rul}')




if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # scheduler 相关
        parser.add_argument('--epochs', type=int, default=1, help='epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
        parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
        parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

        parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
        parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')

        parser.add_argument('--save_folder', type=str, default=None, help='save folder')
        parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')

        return parser.parse_args()


    args = get_args()
    pinn = PINN(args)
    print(pinn.solution_u)
    count_parameters(pinn.solution_u)
    print(pinn.dynamical_F)




