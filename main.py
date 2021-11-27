import argparse
import os
from os import path

from network.inception_resnet_v1 import InceptionResnetV1
from network.fc_layers import Identity
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from network.TorchUtils import TorchModel
from utils.callbacks import DefaultModelCallback, TensorBoardCallback
from utils.utils import register_logger, get_torch_device

from dataloader import get_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIV6 Face Parser")

    # io
    parser.add_argument('--inputs_path', default='features',
                        help="path to inputs")
    parser.add_argument('--log_file', type=str, default="log.log",
                        help="set logging file.")
    parser.add_argument('--exps_dir', type=str, default="exps",
                        help="path to the directory where models and tensorboard would be saved.")
    parser.add_argument('--checkpoint', type=str,
                        help="load a model for resume training")

    # optimization
    # parser.add_argument('--batch_size', type=int, default=60,
    #                     help="batch size")
    parser.add_argument('--save_every', type=int, default=1,
                        help="epochs interval for saving the model checkpoints")
    parser.add_argument('--lr_base', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of training epochs")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    register_logger()
    os.makedirs(args.exps_dir, exist_ok=True)
    models_dir = path.join(args.exps_dir, 'models')
    tb_dir = path.join(args.exps_dir, 'tensorboard')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    device = get_torch_device()

    train_loader, test_loader = get_dataloader()

    if args.checkpoint is not None and path.exists(args.checkpoint):
        model = TorchModel.load_model(args.checkpoint)
    else:
        model = InceptionResnetV1()
        param_dict = torch.load("pretrained/pretrained.pth")
        model.load_state_dict(param_dict)

        for param in model.parameters():
            param.requires_grad = False

        model.last_linear = Identity()
        # model.last_linear = nn.Linear(512, 28)
        model = TorchModel(model)


    # print(model)
    # for param in model.model.logits.parameters():
    #      print(param.requires_grad)

    # assert False

    tb_writer = SummaryWriter(log_dir=tb_dir)
    model.register_callback(DefaultModelCallback(visualization_dir=args.exps_dir))
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer))

    model = model.to(device).train()

    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    model.fit(train_iter=train_loader,
              eval_iter=test_loader,
              criterion=criterion,
              optimizer=optimizer,
              epochs=args.epochs,
              network_model_path_base=models_dir,
              save_every=args.save_every,
              evaluate_every=True)


