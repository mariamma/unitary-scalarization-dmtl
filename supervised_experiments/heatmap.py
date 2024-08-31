import argparse
import torch
import json
import numpy as np
import random
import os
import supervised_experiments.model_selector as model_selector
from supervised_experiments.utils import create_logger
import supervised_experiments.datasets as datasets
from tqdm import tqdm


def load_saved_model(models, tasks, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    models['rep'].load_state_dict(state["model_rep"])
    for t in tasks:
        models[t].load_state_dict(state[f"model_{t}"])


def convert_label_toints(labels):
    nih_classes = [ 'Ate', 'Car', 'Eff', 'Inf', 'Mas', 'Nod', 'Pne',
                        'Pnt', 'Con', 'Ede', 'Emp', 'Fib', 'Ple', 
                        'Her', 'Nor', 'Cov']                
    int_labels = []                        
    for x in labels:
        int_labels.append(nih_classes.index(x))
    return int_labels 


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_heatmap(model, val_rep, t, image_name_path, heatmap_folder):
    out_t, _, _ = model[t](val_rep, None)
    
    out_t[0][1].backward(retain_graph=True)
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
    
    image_name_list = image_name_path.split('/')
    image_name = image_name_list[1]
    print("Heatmap : ", heatmap.shape)
    if ".png" in image_name:
        image_name_pt = image_name.replace(".png", ".pt")
    elif ".jpg" in image_name:
        image_name_pt = image_name.replace(".jpg", ".pt")
    elif ".jpeg" in image_name:
        image_name_pt = image_name.replace(".jpeg", ".pt")
    elif ".JPG" in image_name:
        image_name_pt = image_name.replace(".JPG", ".pt")    
    torch.save(heatmap, os.path.join(heatmap_folder, image_name_pt.replace("/","_")))    
    return 


def generate_heatmap(args, random_seed):
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

    nih_labels = args.nih_labels.split("_")
    nih_labels_indices = convert_label_toints(nih_labels)
    tasks = configs[args.dataset]['tasks']
    tasks = [tasks[t] for t in nih_labels_indices]
    print("Tasks : ", tasks)

    model = model_selector.get_model(args.dataset, configs[args.dataset]['tasks'], device=DEVICE,
                         multi_label=args.multi_label)
    
    load_saved_model(model, tasks, args.net_basename, folder=configs["utils"]["model_storage_old"], name=args.model_type)

    test_loader = datasets.get_dataset(args.dataset, args.batch_size, configs,
                                       generator=g, worker_init_fn=seed_worker, train=False,
                                       partial_dataset=args.partial_dataset, nih_labels=nih_labels,
                                       whatsapp_data = True, image_name = True,
                                       covid_img_only = True)        

    
    foldername = args.net_basename
    print("Foldername : {}".format(foldername))
    os.makedirs(os.path.join(args.heatmap_dir, foldername))

    foldername_path = os.path.join(args.heatmap_dir, foldername)
    heatmap_orig_folder = os.path.join(foldername_path, "original/")
    heatmap_whatsapp_folder = os.path.join(foldername_path, "whatsapp/")

    os.makedirs(heatmap_orig_folder)
    os.makedirs(heatmap_whatsapp_folder)

    # Evaluate the model on the test set.
    model['rep'].eval()
    for m in model:
        model[m].eval()

    for batch_val in tqdm(test_loader):
        image_name =  batch_val[0]
        test_images = batch_val[1].to(DEVICE)
        test_images = test_images.requires_grad_(True)    
        corrupt_imgs = batch_val[2].to(DEVICE)
        corrupt_imgs = corrupt_imgs.requires_grad_(True)    
        test_labels = batch_val[3].to(torch.long).to(DEVICE)

        print("Image name : ", image_name)
        print("Test labels : ", test_labels)

        val_rep, _ = model['rep'](test_images, None)
        val_rep_corrupt, _ = model['rep'](corrupt_imgs, None)   

        if test_labels[0][-1] == 1:
            t = '15'
            create_heatmap(model, val_rep, t, image_name[0], heatmap_orig_folder)
            create_heatmap(model, val_rep_corrupt, t, image_name[0], heatmap_whatsapp_folder)
            
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_basename', type=str, default='', help='basename of network (excludes _x_model.pkl)')
    parser.add_argument('--dataset', type=str, default='cov_nih', help='which dataset to use', choices=['celeba', 'mnist'])
    parser.add_argument('--model_type', type=str, default='last', help='best or last model', choices=['best', 'last'])
    parser.add_argument('--random_seed', type=int, default=1, help='Start random seed to employ for the run.')
    parser.add_argument('--config_file', type=str, default="supervised_experiments/configs.json")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--corruption', type=str, default=None, help='corruption')
    parser.add_argument('--severity', type=str, default=None, help='severity')
    parser.add_argument('--multi_label', type=bool, default=True, help='multi_label_flag')
    parser.add_argument('--nih_labels', type=str, default=True, help='NIH labels to be used')
    parser.add_argument('--partial_dataset', type=bool, default=True, help='Use only part of NIH dataset')
    parser.add_argument('--heatmap_dir', type=str, default="/scratch/mariamma/mtl/heatmap/", help='Heatmap directory')
    args = parser.parse_args()
    generate_heatmap(args, args.random_seed)
    return


main()    