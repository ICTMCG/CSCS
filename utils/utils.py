import torch
import numpy as np
import os
import importlib
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
import random
from .logger import Progbar
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Function

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# https://miopas.github.io/2019/04/17/multiple-classification-metrics/

class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x):
        self.lambd=1
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF.apply(x)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def collate_function(data):
    transposed_data = list(zip(*data))
    img1, lab, img2, imgpaths = transposed_data[0], transposed_data[1], transposed_data[2],transposed_data[3]
    img1 = torch.stack(img1, 0)
    lab = torch.stack(lab, 0)
    img2 = torch.stack(img2, 0)
    return img1, lab, img2, imgpaths

def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.Config()


def read_annotations(data_path, act_shuffle=True):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        one_data=line.split('\t')
        one_data[1]=int(one_data[1]) # because this is label
        data.append(one_data)

    if act_shuffle:
        random.shuffle(data)
    return data

def get_train_paths(args):
    train_data_path = os.path.join(args.data_path, args.train_collection, "annotations", args.train_collection + ".txt")
    val_data_path = os.path.join(args.data_path, args.val_collection, "annotations", args.val_collection + ".txt")
    model_dir = os.path.join(args.data_path, args.train_collection, "models", args.val_collection, args.config_name,
                             "run_%s" % args.run_id)
    return [model_dir, train_data_path, val_data_path]

def get_test_paths(args):
    test_data_path = os.path.join(args.data_path, args.test_collection, 'annotations', args.test_collection + ".txt")
    pred_dir = os.path.join(args.data_path, args.test_collection, "pred", args.model_dir.replace('/models/', '/'))
    return test_data_path, pred_dir

def evaluate_multiclass(gt_labels, pred_labels):
    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average='macro')
    recall = recall_score(gt_labels, pred_labels, average='macro')
    recalls = recall_score(gt_labels, pred_labels, average=None)  # 每一类recall返回
    return {'recalls':recalls,'recall':recall,'f1':f1,'acc':acc}

def evaluate(gt_labels, pred_labels, scores):
    n = len(gt_labels)
    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).reshape(-1)
    assert((tn + fp + fn + tp) == n)

    auc = roc_auc_score(gt_labels, scores)
    ap = average_precision_score(gt_labels, scores)
    sen = float(tp) / (tp + fn)
    spe = float(tn) / (tn + fp)
    f1 = 2.0*sen*spe / (sen + spe)
    acc = float(tn + tp) / n
    return {'auc':auc, 'ap':ap, 'sen':sen, 'spe':spe, 'f1':f1, 'acc':acc}


def plot_confusion_matrix(confusion, labels_name, save_path):
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion, cmap=plt.cm.Blues)  # 在特定的窗口上显示图像
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index - 0.3, second_index, confusion[first_index][second_index])
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()


def caculate(pred, true, th):
    pred_ = (pred > th)
    P = np.sum(true)
    TP = np.sum(pred_ * true)
    TN = np.sum(pred_ == true) - TP
    N = np.sum(true == 0)
    FN = P - TP
    FP = N - TN
    return TP, TN, FP, FN


def calculate_pixel_f1(pd, gt):
    '''
    Args:
        pd: a 1-d np array consit of 0.0 and 1.0 for 1 instance
        gt: a 1-d np array consit of 0.0 and 1.0 for 1 instance

    Returns:
        Return pixel-level scores for segmentation
        f1:
        precision:
        recall:

    Raises
    '''
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    return f1, precision, recall


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad




def predict_set_early(generator, dataloader,runtime_params,save_result=False,save_path=None):
    device = runtime_params['device']
    run_type = runtime_params['run_type']
    generator.eval()
    progbar = Progbar(len(dataloader.dataset), stateful_metrics=['run-type'])
    criterion = nn.CrossEntropyLoss()  
    
    with torch.no_grad():
        # try:
        for i, batch in enumerate(dataloader):

            fake_img_batch, _, img_path, label_batch = batch 
            fake_img = fake_img_batch.reshape((-1, 3, fake_img_batch.size(-2), fake_img_batch.size(-1)))
            labels = label_batch.reshape((-1))
            fake_img, labels = [t.to(device) for t in [fake_img,labels]]
            
            noise = generator(fake_img)[1]
            if save_result:
                for index, n in enumerate(noise):
                    os.makedirs(os.path.join(save_path, img_path[index].split('/')[-3], img_path[index].split('/')[-2]),exist_ok=True)
                    vutils.save_image(torch.unsqueeze(n, 0), os.path.join(save_path, img_path[index].split('/')[-3], img_path[index].split('/')[-2],
                                                                        img_path[index].split('/')[-1]), normalize=True)

            early_output = generator(fake_img)[0]
            fake_loss = criterion(early_output, labels)

            if i == 0:
                probs = early_output
                gt_labels = labels
                img_paths = img_path
            else:
                probs = torch.cat([probs, early_output], dim=0)
                gt_labels = torch.cat([gt_labels, labels])
                img_paths += img_path

            progbar.add(fake_img_batch.size(0), values=[('run-type', run_type),('fake_loss',fake_loss.item())])
        # except:
            # print('error')

    gt_labels = gt_labels.cpu().numpy()
    probs = probs.cpu().numpy() 
    pred_labels = np.argmax(probs,axis=1)
    
    return gt_labels,pred_labels,probs,img_paths


def predict_set_post(generator, classifier, dataloader,runtime_params): 
    device = runtime_params['device']
    run_type = runtime_params['run_type']
    classifier.eval()
    generator.eval()
    progbar = Progbar(len(dataloader.dataset), stateful_metrics=['run-type'])
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            fake_img_batch, real_img_batch, _, _, img_path, label_batch, _ = batch
            fake_img = fake_img_batch.reshape((-1, 3, fake_img_batch.size(-2), fake_img_batch.size(-1)))
            real_img = real_img_batch.reshape((-1, 3, real_img_batch.size(-2), real_img_batch.size(-1)))
            labels = label_batch.reshape((-1))
            real_img, fake_img,labels = [t.to(device) for t in [real_img, fake_img,labels]]

            cls_output_E, noise = generator(fake_img)
            loss_G_cls_E = criterion(cls_output_E, labels)
            real2fake_image = noise + real_img
            cls_output_D = classifier(real2fake_image)
            loss_G_cls_D = criterion(cls_output_D, labels)
            g_loss=loss_G_cls_E+loss_G_cls_D

            if i == 0:
                probs = cls_output_D
                gt_labels = labels
                img_paths = img_path
            else:
                probs = torch.cat([probs, cls_output_D], dim=0)
                gt_labels = torch.cat([gt_labels, labels])
                img_paths += img_path

            progbar.add(fake_img_batch.size(0),
                        values=[('run-type', run_type),('loss_G_cls_D', loss_G_cls_D.item()), ('loss_G_cls_E',loss_G_cls_E.item()), ('g_loss',g_loss.item())])


    gt_labels = gt_labels.cpu().numpy()
    probs = probs.cpu().numpy()
    pred_labels = np.argmax(probs,axis=1)

    return gt_labels,pred_labels,probs,img_paths


