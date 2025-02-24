# coding=utf-8
from __future__ import print_function
import math
from random import random
from random import seed
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import models.ResNet as ResNet
from models.autoaugment import CIFAR10Policy
from models.wrn import WideResNet

from Utils.display_results import get_measures, print_measures, print_measures_with_std
import Utils.score_calculation as lib
from PIL import ImageFile

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def train(model,train_loader, optimizer,scheduler,epoch):
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, total=len(train_loader))
    for data,target in loop:
        data, target = data.cuda(), target.cuda()
        model = model.cuda()
        optimizer.zero_grad()
        prediction = model(data)
        critetion = nn.CrossEntropyLoss()
        loss = critetion(prediction,target)
        pred = prediction.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_description(f'Epoch [{epoch}/{args.training_epochs}]')
        total_loss /= len(train_loader.dataset)
        accuracy = 100. * correct / total
        loop.set_postfix(loss=loss.item(), acc=accuracy)
    scheduler.step(total_loss)



def valid(model, valid_loader):
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.cuda(), target.cuda()
            model = model.cuda()
            prediction = model(data)
            critetion = nn.CrossEntropyLoss()
            loss = critetion(prediction,target)
            valid_loss += loss.item()
            pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    valid_loss /= len(valid_loader.dataset)
    accuracy = 100. * correct / len(valid_loader.dataset)
    return valid_loss, correct, accuracy

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(train_loader,model):
    valid_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            prediction,penultimate = model(data,need_penultimate=4)
            pred = prediction.argmax(dim=1,keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(train_loader.dataset)
    print(correct," ",accuracy)
    return 0

def get_threshold(p=0.9):
    tempres = []
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            _,penultimate = net(data,need_penultimate=args.need_penultimate)
            for i in range(penultimate.size(0)):
                cur_feature = penultimate[i].detach().tolist()
                tempres.extend(cur_feature)
    tempres.sort()
    index = int(len(tempres)*p)
    threshold = tempres[index]
    return threshold


def simple_compute_score_HE(prediction,penultimate,need_mask=False):

    numclass = args.num_class
    #----------------------------------------Step 1: classifier the test feature-----------------------------------
    pred = prediction.argmax(dim=1, keepdim=False)
    pred = pred.cpu().tolist()
    
    #----------------------------------------Step 2: get the stored pattern------------------------------------

    total_stored_feature = None
    for i in range(numclass):
        path = './stored_pattern/avg_stored_pattern/size_{}/{}/{}/stored_avg_class_{}.pth'.format(args.resize_val,args.dataset,args.model,i)
        stored_tensor = torch.load(path)
        if total_stored_feature is None:
            total_stored_feature = stored_tensor
        else:
            total_stored_feature = torch.cat((total_stored_feature,stored_tensor),dim=0)
    #--------------------------------------------------------------------------------------

    target = total_stored_feature[pred,:]
    res = []
    #print(penultimate.shape)
    #for i in range(penultimate.shape[0]):
    #    penultimate[i] = 5*penultimate[i]/torch.norm(penultimate[i])
    # for ablation exp: different metric for SHE
    if args.metric == 'inner_product':
        res_energy_score = torch.sum(torch.mul(penultimate,target),dim=1) #inner product
    elif args.metric == 'euclidean_distance':
        res_energy_score = -torch.sqrt(torch.sum((penultimate-target)**2, dim=1))
    elif args.metric == 'cos_similarity':
        res_energy_score = torch.cosine_similarity(penultimate,target, dim=1)
    lse_res = -to_np(res_energy_score)
    res.append(lse_res)
    return res

def compute_score_HE(prediction,penultimate):
    #----------------------------------------Step 1: classifier the test feature-----------------------------------
    numclass = args.num_class
    feature_list = [None for i in range(numclass)]
    pred = prediction.argmax(dim=1, keepdim=True)
    # get each class tensor
    for i in range(numclass):
        each_label_tensor = torch.tensor([i for _ in range(prediction.size(0))]).cuda()
        target_index = pred.eq(each_label_tensor.view_as(pred))

    # get the penultimate layer
        each_label_feature = penultimate[target_index.squeeze(1)]
        if each_label_feature is None: continue
        if feature_list[i] is None:
            feature_list[i] = each_label_feature
        else:
            feature_list[i] = torch.cat((feature_list[i],each_label_feature),dim=0)
    

    #----------------------------------------Step 2: get the stored pattern------------------------------------
    stored_feature_list = []
    for i in range(numclass):
        path = './stored_pattern/all_stored_pattern/size_{}/{}/{}/stored_all_class_{}.pth'.format(args.resize_val,args.dataset,args.model,i)
        stored_tensor = torch.load(path)
        if stored_tensor is None:
            print(path)
        stored_feature_list.append(stored_tensor) #Here we get all the stored pattestr(i) +'.pth'rns

    res = []
    #----------------------------------------Step 3: compute energy--------------------------------------------------------------------
    for i in range(numclass):

        test_feature = feature_list[i].transpose(0,1) #[dim,B_test]
        stored_feature = stored_feature_list[i] #[B_stored,dim]
        

        if test_feature is None: 
            #print("test is none")
            continue
        if stored_feature is None:
            #print("store is none")
            continue
        res_energy_score = torch.mm(stored_feature,test_feature) #[B_stored,B_test]
        lse_res = -to_np(torch.logsumexp(res_energy_score*args.beita, dim=0)) #[1,B_test]
        res.append(lse_res)
    return res


def get_ood_scores(loader,ood_num_examples,net,in_dist=False):
    _score = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.batch_size and in_dist is False:
                break

            data = data.cuda()


            if args.score == 'SHE':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                _score.extend(simple_compute_score_HE(prediction=output,penultimate=penultimate))
            elif args.score == 'SHE_react':
                output,penultimate = net(data,threshold=args.threshold,need_penultimate=args.need_penultimate)
                _score.extend(simple_compute_score_HE(prediction=output,penultimate=penultimate))
            elif args.score == 'HE':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                _score.extend(compute_score_HE(prediction=output,penultimate=penultimate))
            elif args.score == 'MSP':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                smax = to_np(F.softmax(output, dim=1))
                _score.append(-np.max(smax, axis=1))
            elif args.score == 'Energy':
                output,penultimate = net(data,need_penultimate=args.need_penultimate)
                _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
            elif args.score == 'ReAct':
                output,penultimate = net(data,need_penultimate=args.need_penultimate,threshold=args.threshold)
                _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
        if in_dist:
            return concat(_score).copy()
        else:
            return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader,net,ood_num_examples,args,num_to_avg,in_score):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'SHE_with_perturbation':
            out_score = lib.get_ood_scores_perturbation(args,ood_loader, net, args.batch_size, ood_num_examples, args.T, args.noise)
        else:
            out_score = get_ood_scores(ood_loader,ood_num_examples,net)
        measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])


    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, method_name='method:{}\tsize:{}\tdataset:{}\tmodel:{}'.format(args.score,args.resize_val,args.dataset,args.model))
    else:
        print_measures(auroc, aupr, fpr, method_name='method:{}_dataset:{}'.format(args.score,args.dataset))
    return 100*np.mean(fprs), 100*np.mean(aurocs)



def main():

    # Set random seed
    random_seed = args.random_seed
    seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set training config
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    learning_rate = args.learning_rate
 
    transform_train = transforms.Compose([
        transforms.Resize((args.resize_val,args.resize_val)),
        CIFAR10Policy(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.resize_val,args.resize_val)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if args.dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)
        num_class = 10
        args.num_class=10
    elif args.dataset=='cifar100':
        trainset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)
        num_class = 100
        args.num_class=100

    if args.model == 'resnet18':
        model = ResNet.ResNet18(num_classes=num_class)
        net = ResNet.ResNet18(num_classes=num_class)
        args.resize_val = 112
    elif args.model == 'resnet34':
        model = ResNet.ResNet34(num_classes=num_class)
        net = ResNet.ResNet34(num_classes=num_class)
        args.resize_val = 112
    elif args.model=='wrn':
        model = WideResNet(args.layers, num_class, args.widen_factor, dropRate=args.droprate)
        net = WideResNet(args.layers, num_class, args.widen_factor, dropRate=args.droprate)
        args.resize_val = 64
        
    if len(args.parallel_list)>0:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model).cuda()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94,verbose=True,patience = 1,min_lr = 0.000001)

    #best_acc = 0
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(os.path.join(save_path,'test_useresize_{}_size_{}.pth'.format(args.model,args.resize_val)),map_location=None))
    eval(train_loader,net)
    ood_num_examples = len(validset) // 5
    expected_ap = ood_num_examples / (ood_num_examples + len(validset))
    if args.score == 'SHE_with_perturbation':
        in_score = lib.get_ood_scores_perturbation(args,valid_loader, net, args.batch_size, ood_num_examples, args.T, args.noise, in_dist=True)
    else:
        in_score = get_ood_scores(valid_loader,ood_num_examples,net,in_dist=True)
    fprlist,auclist = [],[]
    # # # /////////////// SVHN /////////////// 
    ood_data = torchvision.datasets.SVHN(root=os.path.join(args.dataset_path,'svhn'), split="test",download=True,transform=transform_test)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\nSVHN Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    
    # # /////////////// LSUN-C ///////////////
    ood_data = dset.ImageFolder(root=os.path.join(args.dataset_path,'LSUN_C'),
                                transform=transform_test)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\nLSUN_C Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    # # # /////////////// LSUN-R ///////////////
    ood_data = dset.ImageFolder(os.path.join(args.dataset_path,'LSUN_resize'),
                                transform=transform_test)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\nLSUN_Resize Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    
    # # /////////////// iSUN ///////////////
    ood_data = dset.ImageFolder(root=os.path.join(args.dataset_path,'iSUN'),
                                transform=transform_test)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\niSUN Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score) 
    fprlist.append(fpr)
    auclist.append(auc)
    
    # # /////////////// Places365 ///////////////
    
    ood_data = dset.ImageFolder(root=os.path.join(args.dataset_path,'Places'),
                                transform=transform_test)                                                   
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\nPlaces365 Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    
    
    
    
    # # /////////////// Textures ///////////////
    
    ood_data = dset.ImageFolder(root=os.path.join(args.dataset_path,'dtd/images'),
                                transform=transform_test)                                                   
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\nTexture Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    
    # # # /////////////// Tiny Imagenet /////////////// # cropped and no sampling of the test set
    ood_data = dset.ImageFolder(root=os.path.join(args.dataset_path,'Imagenet_resize'),
                                transform=transform_test)                                                   
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\nTiny Imagenet Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    
    # /////////////// SUN /////////////// # cropped and no sampling of the test set
    ood_data = dset.ImageFolder(root=os.path.join(args.dataset_path,'SUN'),
                                transform=transform_test)                                                   
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\nSUN Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    
    # # /////////////// iNaturalist /////////////// # cropped and no sampling of the test set
    ood_data = dset.ImageFolder(root=os.path.join(args.dataset_path,'iNaturalist/'),
                                transform=transform_test)                                                   
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    print('\n\niNaturalist  Detection')
    fpr,auc = get_and_print_results(ood_loader,net=net,ood_num_examples=ood_num_examples,args=args,num_to_avg=args.num_to_avg,in_score=in_score)
    fprlist.append(fpr)
    auclist.append(auc)
    
    print('avg:',sum(fprlist)/len(fprlist),sum(auclist)/len(auclist))
    return 0





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='input learning rate for training (default: 0.2)')
    parser.add_argument('--training_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--init_model_with_bias', type=int, default=1,
                        help='init model with bias as 0 or constant positive value')
    parser.add_argument('--random_seed', type=int, default=12,
                        help='input random seed for training (default: 1)')
    parser.add_argument('--model', type=str, default='wrn')
    parser.add_argument('--resize_val', type=int, default=64)
    parser.add_argument('--dataset_path', type=str, default='/data/ood_detection/data/')
    parser.add_argument('--dataset', type=str, default='cifar10',help='ID dataset')
    parser.add_argument('--parallel_list', type=str, default='0',help='give number if want parallel')
    
    #for wrn
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    parser.add_argument('--beita', default=0.01, type=float, help='for HE')
    parser.add_argument('--noise', type=float, default=0.0014, help='pertubation')
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--T', default=1.0, type=float)
    parser.add_argument('--k', default=0.8, type=float)
    parser.add_argument('--metric', type=str, default='inner_product',help='ablation: choose which metric for the SHE')
    parser.add_argument('--score', default='SHE', type=str, help='score options: MSP|Energy|ReAct|HE|SHE|SHE_react|SHE_with_perturbation')
    parser.add_argument('--need_penultimate', default=4, type=int,help='choose which layer as the pattern')

    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')
    
    args = parser.parse_args()
    args.beita = 0.2 if args.model == 'wrn' else 0.01
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.parallel_list
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
            
    save_path = './checkpoints/{}/'.format(args.dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main()
    
