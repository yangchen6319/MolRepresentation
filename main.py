import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataloader.image_dataloader import SmilesImageDataset, load_filenames_and_labels_multitask, get_datasets
from model.model import ClipMol
from model.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from model.train_utils import fix_train_random_seed, load_smiles
from utils.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from utils.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol')

    # basic
    parser.add_argument('--dataset', type=str, default="BBBP", help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='data root')
    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # optimizer
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced'],
                        help='regularization of classification loss')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--image_aug', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')
    parser.add_argument('--save_finetune_ckpt', type=int, default=1, choices=[0, 1],
                        help='1 represents saving best ckpt, 0 represents no saving best ckpt')

    # log
    parser.add_argument('--log_dir', default='./logs/finetune/', help='path to log')

    return parser.parse_args()

def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")
    args.verbose = True

    device, device_ids = setup_device(args.ngpu)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # architecture name
    if args.verbose:
        print('Architecture: {}'.format(args.image_model))    
    
    # initial parameter
    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
    else:
        raise Exception("{} is not supported".format(args.task_type))

    print("eval_metric: {}".format(eval_metric))

    # load data
    names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
    names, labels = np.array(names), np.array(labels)
    print(labels)
    num_tasks = labels.shape[1]
    print("num_tasks: {}".format(num_tasks))
    train_idx, val_idx, test_idx = split_train_val_test_idx(list(range(0, len(names))), frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=args.seed)
    
    name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
    # transform && normalize
    img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Dataset && Dataloader
    train_dataset = SmilesImageDataset(name_train, labels_train, img_transformer=transforms.Compose(img_transformer_train),
                                 normalize=normalize, args=args)
    val_dataset = SmilesImageDataset(name_val, labels_val, img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize, args=args)
    test_dataset = SmilesImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, args=args)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
    
    # load model parameter
    charset_size = train_dataset.get_charset()
    # load model
    model = ClipMol(args, num_tasks, charset_size, 256, 2, 0.2, 0.8)
    model = model.cuda()
    
    # initialize optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.weight_decay,
    )
    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))
    
    # train
    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value,
               'final_train_dict': None,
               'final_val_dict': None,
               'final_test_dict': None}

    early_stop = 0
    patience = 30
    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_dataloader, criterion=criterion,
                                  device=device, epoch=epoch, task_type=args.task_type)
        # evaluate
        train_loss, train_results, train_data_dict = evaluate_on_multitask(model=model, data_loader=train_dataloader,
                                                                           criterion=criterion, device=device,
                                                                           epoch=epoch,
                                                                           task_type=args.task_type,
                                                                           return_data_dict=True)
        val_loss, val_results, val_data_dict = evaluate_on_multitask(model=model, data_loader=val_dataloader,
                                                                     criterion=criterion, device=device,
                                                                     epoch=epoch, task_type=args.task_type,
                                                                     return_data_dict=True)
        test_loss, test_results, test_data_dict = evaluate_on_multitask(model=model, data_loader=test_dataloader,
                                                                        criterion=criterion, device=device, epoch=epoch,
                                                                        task_type=args.task_type, return_data_dict=True)

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        if train_result > results['highest_train']:
            results['highest_train'] = train_result

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            results['final_train_dict'] = train_data_dict
            results['final_val_dict'] = val_data_dict
            results['final_test_dict'] = test_data_dict

        print({"epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train': train_result,
               'Validation': valid_result, 'Test': test_result})
        
    # 保存实验效果最好的数据
    with open('./result/sars-cov-2/'+args.dataset + '_' + args.split + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("final results: highest_valid: {:.3f}, final_train: {:.3f}, final_test: {:.3f}"
          .format(results["highest_valid"], results["final_train"], results["final_test"]))



if __name__ == "__main__":
    args = parse_args()
    main(args)