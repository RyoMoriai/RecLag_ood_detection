from random import seed
import numpy as np
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from models import ResNet 
from Utils.display_results import get_measures, print_measures, print_measures_with_std
import Utils.score_calculation as lib
from PIL import ImageFile
import matplotlib.pyplot as plt

import math
import tensorflow as tf
from tensorflow.keras import Input, models, layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import initializers

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import random


ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name.')
parser.add_argument('--stored_data_path', type=str, default='/data/ood_detection/data/', help='the path for storing data.')
parser.add_argument('--score', default='HE', type=str, help='score options: MSP|Energy|ReAct|HE|SHE|SHE_react|SHE_with_perturbation')
parser.add_argument('--parallel_list', type=str, default='0',help='give number if want parallel')
parser.add_argument('--model', type=str, default='resnet18')

parser.add_argument('--resize_val', default=112, type=int, help='transform resize length')
parser.add_argument('--beita', default=1, type=float, help='for HE')
parser.add_argument('--noise', type=float, default=0.0014, help='pertubation')
parser.add_argument('--threshold', type=float, default=1.0)
parser.add_argument('--T', default=1.0, type=float)
parser.add_argument('--k', default=0.8, type=float)
parser.add_argument('--metric', type=str, default='inner_product',help='ablation: choose which metric for the SHE')

#parameters for wrn
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

parser.add_argument('--need_penultimate', default=4, type=int,help='choose which layer as the pattern')

parser.add_argument('--read-mode', default=0, type=int,help='choose model read or not')
parser.add_argument('--epoch', default=1000, type=int,help='epoch number')
parser.add_argument('--hidden', default=100, type=int,help='hidden layers')
parser.add_argument('--kmeans', default=0, type=int,help='use kmeans or not')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
cudnn.benchmark = True
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform_all = trn.Compose([
    trn.Resize((112,112)),
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

trainset = torchvision.datasets.CIFAR10("oodData", train=True, download=True, transform=transform_all)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

test_data = torchvision.datasets.CIFAR10("oodData", train=False, download=True, transform=transform_all)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
num_class = 10
#print(test_data)
#ood_data = torchvision.datasets.SVHN(root=os.path.join("oodData",'svhn'), split="test",download=True,transform=transform_all)
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

#LSUN_c
#ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'LSUN_C'),transform=transform_all)
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)

#LSUN_R
#ood_data = dset.ImageFolder(os.path.join(args.stored_data_path,'LSUN_resize'),transform=transform_all)
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)

#iSUN
#ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'iSUN'),transform=transform_all)
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)

# # /////////////// Places365 ///////////////
#ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Places'),transform=transform_all) 
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

# # /////////////// Textures ///////////////

#ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'dtd/images'),transform=transform_all)
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)

# # # /////////////// Tiny Imagenet /////////////// # cropped and no sampling of the test set
#ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Imagenet_resize'),transform=transform_all)
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)

# /////////////// SUN /////////////// # cropped and no sampling of the test set
#ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'SUN'),transform=transform_all)   

##  iNaturalist
#ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'iNaturalist/'),transform=transform_all) 
#ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)

PATH = './checkpoints/cifar10/test_useresize_resnet18_size_112.pth'
net = ResNet.ResNet18(num_classes=10)
if args.model == "resnet34":
    PATH = './checkpoints/cifar10/test_useresize_resnet34_size_112.pth'
    net = ResNet.ResNet34(num_classes=10)
if args.model == "wrn":
    PATH = './checkpoints/cifar10/test_useresize_wrn_size_64.pth'
    args.resize_val = 64
    net =  WideResNet(args.layers, args.num_class, args.widen_factor, dropRate=args.droprate)
net = nn.DataParallel(net).cuda()
net.load_state_dict(torch.load(PATH,map_location=None))
net.eval()

to_np = lambda x: x.data.cpu().numpy()
alp = 19750
beita = 0.05




def gauss(x, m, var=1.):  #var=0.1
    coeff = tf.math.rsqrt(2*math.pi*var)
    exp = tf.math.exp(-tf.reduce_sum((x - m)**2, axis=-1, keepdims=False)/(2*var))
    return coeff*exp #, coeff, exp


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.dense = layers.Dense(hidden_dim, activation=None,kernel_initializer=initializers.Zeros(),use_bias=False) # 'relu'

    def call(self, x):
        h = self.dense(x)
        z = tf.nn.softmax(h*args.beita)
        return h, z  # shape=(Batch, hidden_dim)

class Sampler(tf.keras.layers.Layer):
    def __init__(self, n_sample, hidden_dim):
        super(Sampler, self).__init__()
        self.n_sample = n_sample
        self.hidden_dim = hidden_dim

    def call(self, z):
        p = z
        sample = tf.reshape(tf.random.categorical(p, num_samples=self.n_sample), [self.n_sample, z.shape[0]])
        onehot = tf.one_hot(sample, depth=self.hidden_dim)
        return onehot  # shape=(n_sample, Batch, hidden_dim)


class SAE(Model):
    def __init__(self, input_dim, hidden_dim, n_sample):
        super(SAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        #my_tensor = tf.zeros(shape=[input_dim,hidden_dim])
        #my_variable = tf.Variable(my_tensor,trainable=True)
        #print(self.trainable_variables)
        #print(self.trainable_variables.append(my_variable))
        #print(self.trainable_variables)
        self.sampler = Sampler(n_sample,  hidden_dim)
        #self.decoder = Decoder(input_dim, hidden_dim)
        #self.decoder.set_weights(self.encoder.weights[0].T)
        #self.sampler = self.encoder.weights[0]

    def call(self, x):
        h, z   = self.encoder(x)
        #h = tf.tensordot(x, self.trainable_variables, axes=[2, 2])
        #z = tf.nn.softmax(h)
        onehot = self.sampler(z)
        #x_decoded = self.decoder(onehot)
        x_decoded = tf.tensordot(onehot, self.encoder.weights[0], axes=[2, 1])
        #x_decoded = tf.tensordot(onehot,self.trainable_variables, axes=[2, 1])

        temp = gauss(tf.expand_dims(x, axis=0), x_decoded)  # shape=(n_sample, Batch)
        weight = temp / tf.reduce_mean(temp, axis=0)
        #weight = tf.Variable(weight, trainable=False)
        #print(tf.expand_dims(x, axis=0))

        return h, z, onehot, x_decoded, weight

def compute_mat_MHN(stored_tensor):
    h_dim = args.hidden
    data_feature = stored_tensor
    if args.kmeans == 0:
        print(stored_tensor.shape,stored_tensor.shape[0],stored_tensor.shape[1])
        if stored_tensor.shape[0] > h_dim:
            stored_feature = stored_tensor[random.sample(range(stored_tensor.shape[0]), k=h_dim)] #[B_stored,dim]
        else:
            h_dim = stored_tensor.shape[0]
            stored_feature = stored_tensor[:h_dim]
            print(stored_feature.shape,h_dim)
        stored_feature= stored_feature.cpu().numpy()
    else:
        kmeans = KMeans(n_clusters=h_dim, init='k-means++',random_state=0).fit(stored_tensor.cpu().numpy())
        stored_feature=kmeans.cluster_centers_
    print(data_feature.shape)
    print(stored_feature.shape)
    for i in range(h_dim):
        stored_feature[i] = 10*stored_feature[i]/np.linalg.norm(stored_feature[i])
    #alp = stored_feature.norm(dim=0).mean(dim=0)
    #alp = alp*alp*0.2
    image_size = 512
    n_sample = 5 #10

    slse = SAE(input_dim=image_size,
                hidden_dim=h_dim,
                n_sample=n_sample)
    slse.build(input_shape=stored_feature.shape)
    slse.summary()
        
    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    slse.encoder.set_weights([stored_feature.T])
    #print(slse.trainable_variables)
        
    BATCH_SIZE = 500 # n_patterns #256
    dataset = tf.data.Dataset.from_tensor_slices((data_feature.cpu().numpy()))#.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    train_loss = []
    EPOCHS = args.epoch
    for epoch in range(EPOCHS):
        for batch, x in enumerate(dataset):
            batch_loss = train_step(x,slse,optimizer)

        train_loss.append(batch_loss)
        if batch_loss>0.1:
            weight_er = slse.layers[0].weights[0].numpy().T
        if (epoch+1)%20==1:
            print(f'Epoch {epoch+1}, Batch {batch}, Train Loss {train_loss[epoch]}')  # Loss {}, batch_loss.numpy()
    return  slse.encoder.weights[0].numpy().T

def loss_function(x, predict, z, onehot, weight):
    mse = losses.MSE(x, predict)  # shape=(n_sample, Batch)
    eps = 1e-12
    logsoftmax = tf.math.log(tf.reduce_sum(z*onehot, axis=-1) + eps)  # shape=(n_sample, Batch)
    loss = tf.reduce_mean(weight * (mse - logsoftmax))  # mean through n_sample & Batch
    if np.isnan(loss):
        #print(weight,np.linalg.norm(weight),weight.shape)
        loss = 0
    return loss

def train_step(x,slse,optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        h, z, onehot, x_reconstructed, weight = slse(x, training=True)
        weight = tf.Variable(weight, trainable=False)
        loss += loss_function(x, x_reconstructed, z, onehot, weight)
    batch_loss = (loss / len(x))
    variables = slse.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def compute_score_MHN(prediction,penultimate,stored_feature_list,need_mask=False):

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
            feature_list[i] = torch.cat((feature_list[i],each_label_feature),dim=1)
    #----------------------------------------Step 2: get the stored pattern------------------------------------
    res = []
    #----------------------------------------Step 3: compute energy--------------------------------------------------------------------
    norm_list=[]
    for i in range(numclass):
        testnum = feature_list[i].size()[0]
        for j in range(testnum):
            norm_list.append(to_np(torch.norm(feature_list[i][j])))
            feature_list[i][j] = 1*feature_list[i][j]/torch.norm(feature_list[i][j])
        test_feature = torch.t(feature_list[i]) #[dim,B_test]
        stored_feature = torch.from_numpy(stored_feature_list[i].astype(np.float32)).cuda() #[B_stored,dim]
        if test_feature is None: continue
        res_energy_score = torch.mm(stored_feature,test_feature) #[B_stored,B_test]
        sm = to_np(torch.softmax(res_energy_score*args.beita, dim=0))
        lse_res = to_np(torch.logsumexp(res_energy_score*args.beita, dim=0)) #[1,B_test]
        #print(sm[:,0].shape)
        #print(sm[:,0])
        #if not res:
            #print(lse_res[0])
        res.extend(lse_res)
    #print(np.array(norm_list).mean(),np.array(res).mean())
    return res


ood_data = torchvision.datasets.SVHN(root=os.path.join("oodData",'svhn'), split="test",download=True,transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)


ID_score = []
OOD_score = []
first = True
stored_feature_list = []
total_stored_feature = None

if args.read_mode == 0:
    for i in range(10):
        path = './stored_pattern/all_stored_pattern/size_{}/{}/{}/stored_all_class_{}.pth'.format(args.resize_val,args.dataset,args.model,i)
        stored_tensor = torch.load(path)
        MHN_tensor=compute_mat_MHN(stored_tensor)
        stored_feature_list.append(MHN_tensor) #Here we get all the stored pattestr(i) +'.pth'rns
        print(MHN_tensor.shape)
        dir_path = './stored_pattern/MHN_mat/size_{}/{}/{}'.format(args.resize_val,args.dataset,args.model)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(MHN_tensor,'./stored_pattern/MHN_mat/size_{}/{}/{}/stored_all_class_{}_epoch_{}_hidden_{}.pth'.format(args.resize_val,args.dataset,args.model,i,args.epoch,args.hidden))
else:
    for i in range(10):
        path = './stored_pattern/MHN_mat/size_{}/{}/{}/stored_all_class_{}_epoch_{}_hidden_{}.pth'.format(args.resize_val,args.dataset,args.model,i,args.epoch,args.hidden)
        MHN_tensor = torch.load(path)
        stored_feature_list.append(MHN_tensor)
        print(MHN_tensor.shape)

with torch.no_grad():
    for data, target in test_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        ID_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
            
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#plt.plot(fpr, tpr, marker='o')
#plt.xlabel('FPR: False positive rate')
#plt.ylabel('TPR: True positive rate')
#plt.grid()
#plt.savefig('data/sklearn_roc_curve.png')

#print(fpr)
print("SVHN")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))

#LSUN_C
ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'LSUN_C'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score

fig, ax = plt.subplots()
bp = ax.boxplot([ID_score,OOD_score])
plt.show()

tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#plt.plot(fpr, tpr, marker='o')
#plt.xlabel('FPR: False positive rate')
#plt.ylabel('TPR: True positive rate')
#plt.grid()
#plt.savefig('data/sklearn_roc_curve.png')

#print(fpr)
print("LSUN_C")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))

#LSUN_R
ood_data = dset.ImageFolder(os.path.join(args.stored_data_path,'LSUN_resize'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score

fig2, ax2 = plt.subplots()
bp2 = ax2.boxplot([ID_score,OOD_score])
plt.show()

tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#plt.plot(fpr, tpr, marker='o')
#plt.xlabel('FPR: False positive rate')
#plt.ylabel('TPR: True positive rate')
#plt.grid()
#plt.savefig('data/sklearn_roc_curve.png')

#print(fpr)
print("LSUN_R")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))



ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'iSUN'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#print(fpr)
print("iSUN")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))


ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Places'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#print(fpr)
print("Places")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))

ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Places'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#print(fpr)
print("Places")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))

ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'dtd/images'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#print(fpr)
print("DTD")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))

ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'Imagenet_resize'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#print(fpr)
print("Tiny Imagenet")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))


ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'SUN'),transform=transform_all)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#print(fpr)
print("SUN")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))


ood_data = dset.ImageFolder(root=os.path.join(args.stored_data_path,'iNaturalist/'),transform=transform_all) 
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
OOD_score = []
with torch.no_grad():
    for data, target in ood_loader:
        
        data = data.cuda()
        output,penultimate = net(data,need_penultimate=4)
        OOD_score.extend(compute_score_MHN(prediction=output,penultimate=penultimate,stored_feature_list=stored_feature_list))
#print(len(OOD_score),len(ID_score))
minScore = min(ID_score)
maxScore = max(OOD_score)
score_list = ID_score+OOD_score
tf_list = [1 for i in range(len(ID_score))] + [0 for i in range(len(OOD_score))]
score_np = np.array(score_list)
tf_np = np.array(tf_list)

fpr, tpr, thresholds = roc_curve(tf_np, score_np)

#print(fpr)
print("iNaturalist")
for i in range(len(fpr)):
    if tpr[i] > 0.95:
        print("FPR95:",fpr[i])
        break

print("AUROC:",roc_auc_score(tf_list, score_list))


#for i in range(500):
#    thr = i*(maxScore - minScore)/500.0 + minScore
#    tp = sum(x > thr for x in ID_score)
#    fn = sum(x < thr for x in ID_score)
#    fp = sum(x > thr for x in OOD_score)
#    tn = sum(x < thr for x in OOD_score)
#    tpr = tp/(tp+fn)
#    fpr = fp/(fp+tn)
#    print(thr,tpr,fpr)
#    if tpr<0.95:
#        print(tpr,fpr)
#        alp=thr
#        break

#tp = sum(x > alp for x in ID_score)
#tn = sum(x < alp for x in ID_score)
#fp = sum(x > alp for x in OOD_score)
#fn = sum(x < alp for x in OOD_score)


#print('{} {}\n{} {}'.format(tp,tn,fp,fn))
