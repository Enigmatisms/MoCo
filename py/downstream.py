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

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from encoder import ResEncoder
from linear import Linear

to_tensor = transforms.ToTensor()

def rightCalc(pred:torch.Tensor, gt:torch.Tensor):
    _, idx = torch.max(pred, dim = -1)
    return sum(idx == gt)

if __name__ == "__main__":
    vec_dim = 128
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 50, help = "Training lasts for . epochs")
    parser.add_argument("--max_iter", type = int, default = 20, help = "max iteration number")
    parser.add_argument("--batch_size", type = int, default = 50, help = "Batch size in training")
    parser.add_argument("--eval_time", type = int, default = 5, help = "Evaluate every <eval_time> times")
    parser.add_argument("--eval_size", type = int, default = 10, help = "Test only part of the images to save time")
    parser.add_argument("--check_point", type = int, default = 200, help = "Save checkpoint file every <> times.")
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
    chpt_time   = args.check_point
    rm_chpt     = args.remove_chpt

    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    if os.path.exists("..\\chpt"):
        if rm_chpt:
            shutil.rmtree("..\\chpt")
            os.mkdir("..\\chpt")
    else:
        os.mkdir("..\\chpt")

    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    train_set = DataLoader(
        datasets.CIFAR10("..\\data\\", 
            train = True, download = False, transform = to_tensor),
        batch_size = batch_size, shuffle = True,
    )

    test_set = DataLoader(
        datasets.CIFAR10("..\\data\\", 
            train = False, download = False, transform = to_tensor),
        batch_size = batch_size, shuffle = True,
    )

    encoder = ResEncoder()
    net = Linear(128, 10)

    if use_cuda and torch.cuda.is_available():
        net = net.cuda()
        encoder = encoder.cuda()
    else:
        print("CUDA is not available.")
        exit(0)

    if args.load_chpt:
        save = torch.load("..\\model\\check_point_128_99.pt")
        save_model = save['model']
        model_dict = net.state_dict()
        state_dict = {k:v for k, v in save_model.items() if k in model_dict}
        model_dict.update(state_dict)
        encoder.load_state_dict(model_dict) 

    opt = optim.Adam(net.parameters(), lr = 0.001)
    encoder = encoder.eval()
    encoder = encoder.requires_grad_(False)
    loss_func = nn.CrossEntropyLoss()
    train_batch_num = len(train_set)
    for i in range(epochs):
        train_rights = 0
        for n, (x, y) in enumerate(train_set):
            opt.zero_grad()
            x = x.cuda()
            y = y.cuda()
            enc = encoder.forward(x)
            out = net.forward(enc)
            loss = loss_func(out, y)
            train_rights += rightCalc(out, y)
            loss.backward()
            opt.step()
            if n % eval_time == 0:
                net.eval()
                with torch.no_grad():
                    test_loss = 0
                    test_cnt = 0
                    test_rights = 0
                    for seq, (x, y) in enumerate(test_set):
                        x = x.cuda()
                        y = y.cuda()
                        enc = encoder.forward(x)
                        out = net.forward(enc)
                        test_loss
                        test_loss += loss_func(out, y).item()
                        test_rights += rightCalc(out, y)
                        test_cnt += 1
                        if seq > args.eval_size:
                            break
                    test_loss /= test_cnt
                    test_acc = test_rights / (batch_size * args.eval_size)
                    train_acc = train_rights / (batch_size * eval_time)
                    train_rights = 0
                    writer.add_scalar('Loss/Test Loss', test_loss, n + train_batch_num * i)
                    writer.add_scalar('Loss/Train Loss', loss.item(), n + train_batch_num * i)
                    writer.add_scalar('Acc/Train Accuracy', train_acc, n + train_batch_num * i)
                    writer.add_scalar('Acc/Test Accuracy', test_acc, n + train_batch_num * i)
                    print("Epoch: %3d / %3d\t Batch %4d / %4d\t train loss: %.4f\t test loss: %.4f\ttrain acc: %.4f\ttest_acc: %.4f"%(
                        i, epochs, n, train_batch_num, loss.item(), test_loss, train_acc, test_acc
                    ))
                net.train()
            if (n + 1) % chpt_time == 0:
                name = "..\\chpt\\downstream_%d_%d.pt"%(i, n)
                torch.save({'model': net.state_dict(), 'optimizer': opt.state_dict()}, name)
    torch.save({
        'model': net.state_dict(),
        'optimizer': opt.state_dict()},
        "..\\model\\baseline.pth"
    )
    writer.close()
    print("Output completed.")
