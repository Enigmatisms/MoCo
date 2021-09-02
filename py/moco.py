#-*-coding=utf-8-*-
"""
    Momentum Contrast Implementation
    This is the instantiated implementation of the pseudo code in the original paper
    @author Sentinel 
    @date 2021.9.1 
"""
import os
import torch
import shutil
import argparse
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.datasets.cifar import CIFAR10
from collections import deque
from PIL import Image

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from encoder import ResEncoder

class CIFAR10Loader(CIFAR10):
    def __init__(self, root, train, transform, target_transform = None, download = False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)
        return im_1, im_2

to_tensor = transforms.ToTensor()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

if __name__ == "__main__":
    vec_dim = 128
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 50, help = "Training lasts for . epochs")
    parser.add_argument("--max_iter", type = int, default = 20, help = "max iteration number")
    parser.add_argument("--batch_size", type = int, default = 50, help = "Batch size in training")
    parser.add_argument("--queue_size", type = int, default = 20, help = "Max queue size in MoCo queue dictionary look up")
    parser.add_argument("--eval_time", type = int, default = 5, help = "Evaluate every <eval_time> times")
    parser.add_argument("--eval_size", type = int, default = 10, help = "Test only part of the images to save time")
    parser.add_argument("--check_point", type = int, default = 200, help = "Save checkpoint file every <> times.")
    parser.add_argument("--momentum", type = float, default = 0.999, help = "Momentum update ratio")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-b", "--use_bn", action = "store_true", help = "Use batch normalization in Encoder")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    parser.add_argument("-r", "--remove_chpt", default = False, action = "store_true", help = "Delete all checkpoints")
    parser.add_argument("-s", "--load_chpt", default = False, action = "store_true", help = "Resume training from a recent check point.")
    args = parser.parse_args()

    epochs      = args.epochs
    del_dir     = args.del_dir
    use_cuda    = args.cuda
    max_iter    = args.max_iter
    use_bn      = args.use_bn
    batch_size  = args.batch_size
    eval_time   = args.eval_time
    mr          = args.momentum
    chpt_time   = args.check_point
    rm_chpt     = args.remove_chpt

    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    if os.path.exists("../chpt"):
        if rm_chpt:
            shutil.rmtree("../chpt")
            os.mkdir("../chpt")
    else:
        os.mkdir("../chpt")

    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    train_set = DataLoader(
        CIFAR10Loader("../data/", 
            train = True, download = False, transform = to_tensor),
        batch_size = batch_size, shuffle = True,
    )

    test_set = DataLoader(
        CIFAR10Loader("../data/", 
            train = False, download = False, transform = to_tensor),
        batch_size = batch_size, shuffle = True,
    )

    f_q = ResEncoder()
    f_k = ResEncoder()
    f_k.load_state_dict(f_q.state_dict())

    if use_cuda and torch.cuda.is_available():
        f_q = f_q.cuda()
        f_k = f_k.cuda()
    else:
        print("CUDA is not available.")
        exit(0)

    deq = deque([torch.normal(0, 1, (batch_size, vec_dim)).cuda()], maxlen = args.queue_size)
    opt = optim.SGD(f_q.parameters(), lr = 0.03, weight_decay = 0.0001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    train_batch_num = len(train_set)
    f_k.eval()
    for i in range(epochs):
        for n, (x_q, x_k) in enumerate(train_set):
            opt.zero_grad()
            x_q = x_q.cuda()
            x_k = x_k.cuda()
            q = f_q.forward(x_q)
            k = f_k.forward(x_k)
            k = k.detach()
            l_pos = torch.bmm(q.view(batch_size, 1, vec_dim), k.view(batch_size, vec_dim, 1)).squeeze(dim = -1)
            lut = torch.cat([_k for _k in deq], dim = 0)      # deq are bunch of (batch_size, vec_dim)
            l_neg = torch.mm(q.view(batch_size, vec_dim), lut.transpose(0, 1))
            logits = torch.cat([l_pos, l_neg], dim = 1)
            labels = torch.zeros(batch_size).long().cuda()
            loss = loss_func(logits, labels)
            loss.backward()
            opt.step()
            f_k.paramUpdate(f_q.parameters(), mr)
            if n % eval_time == 0:
                f_q.eval()
                with torch.no_grad():
                    test_loss = 0
                    test_cnt = 0
                    for seq, (y_q, y_k) in enumerate(test_set):
                        y_q = y_q.cuda()
                        y_k = y_k.cuda()
                        q = f_q.forward(x_q)
                        tk = f_k.forward(x_k)
                        tk = tk.detach()
                        l_pos = torch.bmm(q.view(batch_size, 1, vec_dim), tk.view(batch_size, vec_dim, 1)).squeeze(dim = -1)
                        lut = torch.cat([_k for _k in deq], dim = 0)      # deq are bunch of (batch_size, vec_dim)
                        l_neg = torch.mm(q.view(batch_size, vec_dim), lut.transpose(0, 1))
                        logits = torch.cat([l_pos, l_neg], dim = 1)
                        labels = torch.zeros(batch_size).long().cuda()
                        test_loss += loss_func(logits, labels).item()
                        test_cnt += 1
                        if seq > args.eval_size:
                            break
                    test_loss /= test_cnt
                    writer.add_scalar('Loss/Test Loss', test_loss, test_cnt + train_batch_num * i)
                    writer.add_scalar('Loss/Train Loss', loss.item(), test_cnt + train_batch_num * i)
                    print("Epoch: %3d / %3d\t Batch %4d / %4d\t train loss: %.4f\t test loss: %.4f"%(
                        i, epochs, n, train_batch_num, loss.item(), test_loss,
                    ))
                f_q.train()
            if (n + 1) % chpt_time == 0:
                name = "../chpt/check_point_%d_%d.pt"%(i, n)
                torch.save({'model': f_q.state_dict(), 'optimizer': opt.state_dict()}, name)
            deq.append(k)
        # save_image(gen.detach().clamp_(0, 1), "../imgs/G_%d.jpg"%(epoch + 1), 1)
    torch.save({
        'model': f_q.state_dict(),
        'optimizer': opt.state_dict()},
        "../model/model.pth"
    )
    writer.close()
    print("Output completed.")
