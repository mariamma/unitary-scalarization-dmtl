import argparse
import json
import torch
import random
import numpy as np
import os
import wandb
from supervised_experiments.utils import create_logger

import supervised_experiments.losses as losses_f
import supervised_experiments.datasets as datasets
import supervised_experiments.metrics as metrics
import supervised_experiments.model_selector as model_selector
from torch import linalg as LA
from tqdm import tqdm
from skimage.transform import resize
import pandas as pd

def load_saved_model(models, tasks, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    models['rep'].load_state_dict(state["model_rep"])
    for t in tasks:
        models[t].load_state_dict(state[f"model_{t}"])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_gradient(model, val_rep, task, input_img):
    out_t_val, _, pre_softmax = model[task](val_rep, None)
    gradient = torch.autograd.grad(torch.max(pre_softmax), input_img, retain_graph=True)
    return gradient[0], out_t_val


def convert_label_toints(labels):
    nih_classes = [ 'Ate', 'Car', 'Eff', 'Inf', 'Mas', 'Nod', 'Pne',
                        'Pnt', 'Con', 'Ede', 'Emp', 'Fib', 'Ple', 
                        'Her', 'Nor', 'Cov']                
    int_labels = []                        
    for x in labels:
        int_labels.append(nih_classes.index(x))
    return int_labels 


def evaluate_correaltion(main_tasks, sec_tasks, model, val_rep,test_images, val_rep_corrupt, corrupt_imgs):
    rho_dict = {}
    for idt, t in enumerate(main_tasks):
        gradient_t_orig, out_t_val = get_gradient(model, val_rep, t, test_images)
            
        gradient_t_corr, out_t_val_corrupt = get_gradient(model, val_rep_corrupt, t, corrupt_imgs)

        u_t = (gradient_t_orig - gradient_t_corr)
            # print("t={}, size={}, norm={}".format(t, u_t.size(), LA.vector_norm(torch.flatten(u_t)) ))
        u_t = torch.mean(u_t, [0,1])
            # print("t={}, size={}, norm={}".format(t, u_t.size(), LA.vector_norm(torch.flatten(u_t)) ))

        for ids, s in enumerate(sec_tasks):
            gradient_s_orig, out_s_val = get_gradient(model, val_rep, s, test_images)
            gradient_s_corr, out_s_val_corrupt = get_gradient(model, val_rep_corrupt, s, corrupt_imgs)
            u_s = gradient_s_orig - gradient_s_corr
                # print("s={}, size={}, norm={}".format(s, u_s.size(), LA.vector_norm(torch.flatten(u_s)) ))
            u_s = torch.mean(u_s, [0,1])
                # print("s={}, size={}, norm={}".format(s, u_s.size(), LA.vector_norm(torch.flatten(u_s)) ))

            diff_pdt = torch.dot(torch.flatten(u_t), torch.flatten(u_s))/(LA.vector_norm(torch.flatten(u_t)) * LA.vector_norm(torch.flatten(u_s)))    
            diff_pdt = diff_pdt.cpu().detach().numpy()
            rho_dict["rho_"+t+"_"+s] = diff_pdt
    return rho_dict, gradient_t_orig, gradient_t_corr


def create_heatmap(model, val_rep, t, image_name_path):
    out_t, _, _ = model[t](val_rep, None)
    if out_t[0][1] > out_t[0][0]:
        out_t[0][1].backward(retain_graph=True)
    else:
        out_t[0][0].backward(retain_graph=True)
    gradients = model[t].get_activations_gradient()
    activations = model[t].get_activations(val_rep).detach()
    
    # weight the channels by corresponding gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    # heatmap = np.maximum(heatmap, 0)
    heatmap = torch.nn.functional.relu(heatmap)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    return gradients, heatmap


def ols_score(fname, image_size, lung_segment_path, heatmap_hr,heatmap_lr, row, grad_type):
    lung_region = np.load(os.path.join(lung_segment_path,fname))
    lung_region = resize(lung_region, (image_size, image_size))
        
    heatmap_hr = heatmap_hr.cpu().detach().numpy()
    heatmap_lr = heatmap_lr.cpu().detach().numpy()        
    if np.sum(lung_region) > 0:
        image_name = fname
        for threshold in [0, 0.00001, 0.00002, 0.00003, 0.00005, 0.0001, 0.001]:
            # print("Numerator :", np.sum((heatmap_hr>threshold)), np.sum((lung_region>0)) , np.sum((heatmap_hr>threshold)*(lung_region>0)))
            # print("Denominator :", np.sum(heatmap_hr>threshold)+0.00000001)
            dice_hr = (np.sum((heatmap_hr>threshold)*(lung_region>0))/(np.sum(heatmap_hr>threshold)+0.00000001))
            dice_lr = (np.sum((heatmap_lr>threshold)*(lung_region>0))/(np.sum(heatmap_lr>threshold)+0.00000001))
            row[grad_type+'HR_Score_' + str(threshold)] = dice_hr
            row[grad_type+'LR_Score_' + str(threshold)] = dice_lr
    return row    

def replace_extension(image_name):
    if ".png" in image_name:
        image_name_pt = image_name.replace(".png", ".npy")
    elif ".jpg" in image_name:
        image_name_pt = image_name.replace(".jpg", ".npy")
    elif ".jpeg" in image_name:
        image_name_pt = image_name.replace(".jpeg", ".npy")
    elif ".JPG" in image_name:
        image_name_pt = image_name.replace(".JPG", ".npy")        
    else:
        print(image_name)
    return image_name_pt        


def normalize_gradient(heatmap):
    heatmap = torch.mean(heatmap,1).squeeze()
    heatmap = torch.nn.functional.relu(heatmap)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    return heatmap
            

def test_multi_task(args, random_seed):
    # Set random seeds.
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = create_logger('Main')
    with open('supervised_experiments/configs.json') as config_params:
        configs = json.load(config_params)

    nih_labels = args.task_labels.split("_")
    nih_labels_indices = convert_label_toints(nih_labels)
    tasks = configs[args.dataset]['tasks']
    tasks_total = tasks
    

    tasks = [tasks[t] for t in nih_labels_indices]
    print("Tasks : ", tasks)
          
    data_labels = args.data_labels.split("_")      
    test_loader = datasets.get_dataset(args.dataset, args.batch_size, configs,
                                       generator=g, worker_init_fn=seed_worker, train=False,
                                       partial_dataset=args.partial_dataset, nih_labels=data_labels,
                                       whatsapp_data = True, image_name = True,
                                       covid_img_only = True)                                               

    loss_fn = losses_f.get_loss(args.dataset, configs[args.dataset]['tasks'])
    metric, aggregators, _ = metrics.get_metrics(args.dataset, configs[args.dataset]['tasks'])
    metric_corr, aggregators_corr, _ = metrics.get_metrics(args.dataset, configs[args.dataset]['tasks'])

    model = model_selector.get_model(args.dataset, configs[args.dataset]['tasks'], device=DEVICE)
    
    load_saved_model(model, tasks, args.net_basename, folder=configs["utils"]["model_storage"], name=args.model_type)

    # Evaluate the model on the test set.
    for m in model:
        model[m].eval()

    losses = {t: 0.0 for t in tasks}
    losses_corr = {t: 0.0 for t in tasks}
    num_test_batches = 0
    cos = torch.nn.CosineSimilarity(dim=0)

    df = pd.DataFrame()
    for batch_val in tqdm(test_loader):
        image_name =  batch_val[0]
        test_images = batch_val[1].to(DEVICE)
        test_images = test_images.requires_grad_(True)    
        corrupt_imgs = batch_val[2].to(DEVICE)
        corrupt_imgs = corrupt_imgs.requires_grad_(True)    
        test_labels = batch_val[3].to(torch.long).to(DEVICE)

        val_rep, _ = model['rep'](test_images, None)
        val_rep_corrupt, _ = model['rep'](corrupt_imgs, None)
        
        if test_labels[0][-1] == 1:
            main_task = ['15']
            # gradients_orig, heatmap_orig = create_heatmap(model, val_rep, main_task, image_name[0])
            # gradients_corr, heatmap_corr = create_heatmap(model, val_rep_corrupt, main_task, image_name[0])

            rho_dict, gradient_t_orig, gradient_t_corr = evaluate_correaltion(main_task, tasks, model, \
                    val_rep, test_images, val_rep_corrupt, corrupt_imgs)
            rho_dict['image_name'] = image_name[0]        
            
            cosine_dist = cos(torch.flatten(gradient_t_orig), torch.flatten(gradient_t_corr))
            rho_dict['cosine_sim'] = cosine_dist.cpu().detach().numpy()

            diff_norm = LA.vector_norm(gradient_t_orig - gradient_t_corr)
            rho_dict['sal_diff_norm'] = diff_norm.cpu().detach().numpy()

            gradient_t_orig_sq = gradient_t_orig * gradient_t_orig    
            gradient_t_corr_sq = gradient_t_corr * gradient_t_corr   

            gradient_t_orig = normalize_gradient(gradient_t_orig)
            gradient_t_corr = normalize_gradient(gradient_t_corr)
            rho_dict = ols_score(replace_extension(image_name[0]), gradient_t_orig.shape[0], \
                    args.lung_segment_path, gradient_t_orig, gradient_t_corr, rho_dict, \
                    "Smooth")

            gradient_t_orig_sq = normalize_gradient(gradient_t_orig_sq)
            gradient_t_corr_sq = normalize_gradient(gradient_t_corr_sq)   
            rho_dict = ols_score(replace_extension(image_name[0]), gradient_t_orig.shape[0], \
                    args.lung_segment_path, gradient_t_orig_sq, gradient_t_corr_sq, rho_dict, \
                    "SmoothSq") 
            # print(rho_dict)  
            df = pd.concat([df, pd.DataFrame([rho_dict])], ignore_index=True)
    filename = args.dataset + args.net_basename + ".csv"            
    df.to_csv(os.path.join(configs[args.dataset]['test_results'], filename), index=False)



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_basename', type=str, default='', help='basename of network (excludes _x_model.pkl)')
    parser.add_argument('--dataset', type=str, default='cov_nih', help='which dataset to use', choices=['celeba', 'mnist'])
    parser.add_argument('--model_type', type=str, default='last', help='best or last model', choices=['best', 'last'])
    parser.add_argument('--random_seed', type=int, default=1, help='Start random seed to employ for the run.')
    parser.add_argument('--config_file', type=str, default="supervised_experiments/configs.json")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--corruption', type=str, default=None, help='corruption')
    parser.add_argument('--severity', type=str, default=None, help='severity')
    # parser.add_argument('--nih_labels', type=str, default=True, help='NIH labels to be used')
    parser.add_argument('--task_labels', type=str, default=True, help='NIH labels to be used')
    parser.add_argument('--data_labels', type=str, default=True, help='Dataset labels labels to be used')
    parser.add_argument('--partial_dataset', type=bool, default=True, help='Use only part of NIH dataset')
    parser.add_argument('--lung_segment_path', type=str,
            default='/data6/rajivporana_scratch/datasets/covidx3_upsampled_lung_segment/test/',
            help='Lung segmented image path')
            
    args = parser.parse_args()

    test_multi_task(args, args.random_seed)
