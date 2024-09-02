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

def load_saved_model(models, tasks, net_basename, folder="saved_models/", name="best"):
    state = torch.load(f"{folder}{net_basename}_{name}_model.pkl")
    models['rep'].load_state_dict(state["model_rep"])
    for t in tasks:
        models[t].load_state_dict(state[f"model_{t}"])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def log_test_results(tasks, metric, losses, num_test_batches, aggregators, logger, do_wandb=False,
                     model_name=None, epoch=-1):
    spaced_modelname = model_name + " " if model_name is not None else ""
    underscored_modelname = model_name + "_" if model_name is not None else ""
    clog = ""
    metric_results = {}
    for t in tasks:
        metric_results[t] = metric[t].get_result()
        metric[t].reset()
        clog += ' {}test_loss {} = {:5.4f}'.format(spaced_modelname, t, losses[t] / num_test_batches)
        for metric_key in metric_results[t]:
            clog += ' {}test metric-{} {} = {:5.4f}'.format(spaced_modelname, metric_key, t, metric_results[t][metric_key])
        clog += " |||"

    for agg_key in aggregators.keys():
        clog += ' {}test metric-{} = {:5.4f}'.format(spaced_modelname, agg_key, aggregators[agg_key](metric_results))

    logger.info(clog)
    test_stats = {}
    for i, t in enumerate(tasks):
        test_stats[f"{underscored_modelname}test_loss_{t}"] = losses[t] / num_test_batches
        for metric_key in metric_results[t]:
            test_stats[f"{underscored_modelname}test_metric_{metric_key}_{t}"] = metric_results[t][metric_key]

    for agg_key in aggregators.keys():
        test_stats[f"{underscored_modelname}test_metric_{agg_key}"] = aggregators[agg_key](metric_results)

    if do_wandb:
        wandb.log(test_stats, step=epoch)
    return test_stats


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

    nih_labels = args.nih_labels.split("_")
    nih_labels_indices = convert_label_toints(nih_labels)
    tasks = configs[args.dataset]['tasks']
    tasks_total = tasks
    pos_sum = np.zeros([len(tasks_total), len(tasks_total)])    
    neg_sum = np.zeros([len(tasks_total), len(tasks_total)])    
    pos_cnt = np.zeros([len(tasks_total), len(tasks_total)])    
    neg_cnt = np.zeros([len(tasks_total), len(tasks_total)])

    grad_avg = np.zeros([len(tasks_total), len(tasks_total)])
    grad_sum = np.zeros([len(tasks_total), len(tasks_total)])
    grad_cnt = np.zeros([len(tasks_total), len(tasks_total)])

    gradsum_avg = np.zeros([len(tasks_total), len(tasks_total)])
    gradsum_sum = np.zeros([len(tasks_total), len(tasks_total)])
    gradsum_cnt = np.zeros([len(tasks_total), len(tasks_total)])

    pos_avg = np.zeros([len(tasks), len(tasks)])   
    neg_avg = np.zeros([len(tasks), len(tasks)])

    tasks = [tasks[t] for t in nih_labels_indices]
    print("Tasks : ", tasks)
   
    test_loader = datasets.get_dataset(args.dataset, args.batch_size, configs,
                                       generator=g, worker_init_fn=seed_worker, train=False,
                                       partial_dataset=args.partial_dataset, nih_labels=nih_labels,
                                       whatsapp_data = True)        

    loss_fn = losses_f.get_loss(args.dataset, configs[args.dataset]['tasks'])
    metric, aggregators, _ = metrics.get_metrics(args.dataset, configs[args.dataset]['tasks'])
    metric_corr, aggregators_corr, _ = metrics.get_metrics(args.dataset, configs[args.dataset]['tasks'])

    model = model_selector.get_model(args.dataset, configs[args.dataset]['tasks'], device=DEVICE)
    
    load_saved_model(model, tasks, args.net_basename, folder=configs["utils"]["model_storage_old"], name=args.model_type)

    # Evaluate the model on the test set.
    for m in model:
        model[m].eval()

    losses = {t: 0.0 for t in tasks}
    losses_corr = {t: 0.0 for t in tasks}
    num_test_batches = 0
    
    for batch_val in tqdm(test_loader):
        test_images = batch_val[0].to(DEVICE)
        test_images = test_images.requires_grad_(True)    
        corrupt_imgs = batch_val[1].to(DEVICE)
        corrupt_imgs = corrupt_imgs.requires_grad_(True)    
        test_labels = batch_val[2].to(torch.long).to(DEVICE)

        val_rep, _ = model['rep'](test_images, None)
        val_rep_corrupt, _ = model['rep'](corrupt_imgs, None)
        # print("Images norm : {}".format(LA.vector_norm(torch.flatten(test_images))))
        # print("Corrupt images norm : {}".format(LA.vector_norm(torch.flatten(corrupt_imgs))))
        for idt, t in enumerate(tasks):
            gradient_t_orig, out_t_val = get_gradient(model, val_rep, t, test_images)
            
            loss_t = loss_fn[t](out_t_val, test_labels[:, idt])
            losses[t] += loss_t.item()  # for logging purposes
            metric[t].update(out_t_val, test_labels[:, idt])

            gradient_t_corr, out_t_val_corrupt = get_gradient(model, val_rep_corrupt, t, corrupt_imgs)
            loss_t_corr = loss_fn[t](out_t_val_corrupt, test_labels[:, idt])
            losses_corr[t] += loss_t_corr.item()  # for logging purposes
            metric_corr[t].update(out_t_val_corrupt, test_labels[:, idt])

            u_t = (gradient_t_orig - gradient_t_corr)
            
            # print("t={}, size={}, norm={}".format(t, u_t.size(), LA.vector_norm(torch.flatten(u_t)) ))
            u_t = torch.mean(u_t, [0,1])
            # print("t={}, size={}, norm={}".format(t, u_t.size(), LA.vector_norm(torch.flatten(u_t)) ))

            for ids, s in enumerate(tasks):
                gradient_s_orig, out_s_val = get_gradient(model, val_rep, s, test_images)
                gradient_s_corr, out_s_val_corrupt = get_gradient(model, val_rep_corrupt, s, corrupt_imgs)
                u_s = gradient_s_orig - gradient_s_corr
                
                # print("s={}, size={}, norm={}".format(s, u_s.size(), LA.vector_norm(torch.flatten(u_s)) ))
                u_s = torch.mean(u_s, [0,1])
                # print("s={}, size={}, norm={}".format(s, u_s.size(), LA.vector_norm(torch.flatten(u_s)) ))

                diff_pdt = torch.dot(torch.flatten(u_t), torch.flatten(u_s))/(LA.vector_norm(torch.flatten(u_t)) * LA.vector_norm(torch.flatten(u_s)))    
                diff_pdt_only = torch.dot(torch.flatten(u_t), torch.flatten(u_s))

                # print("{},{} -> {}".format(t, s, diff_pdt))
                t = int(t)
                s = int(s)
                if diff_pdt > 0:
                    pos_sum[t][s] += diff_pdt
                    pos_cnt[t][s] += 1
                else:
                    neg_sum[t][s] += diff_pdt
                    neg_cnt[t][s] += 1                       
                grad_sum[t][s] += diff_pdt
                grad_cnt[t][s] += 1

                gradsum_sum[t][s] += diff_pdt_only
                gradsum_cnt[t][s] += 1
        num_test_batches += 1
    
    for t in tasks:
        for s in tasks:
            t = int(t)
            s = int(s)
            if pos_cnt[t][s] > 0:
                pos_avg[t][s] = pos_sum[t][s]/pos_cnt[t][s]
            if neg_cnt[t][s] > 0:
                neg_avg[t][s] = neg_sum[t][s]/neg_cnt[t][s]
            grad_avg[t][s] = grad_sum[t][s]/(grad_cnt[t][s])
            gradsum_avg[t][s] = gradsum_sum[t][s]/gradsum_cnt[t][s]

    results_folder = configs["utils"]["results_storage"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    np.savetxt(f"{results_folder}{args.net_basename}_{args.model_type}_pos_avg.csv", pos_avg, delimiter=",")
    np.savetxt(f"{results_folder}{args.net_basename}_{args.model_type}_neg_avg.csv", neg_avg, delimiter=",")
    np.savetxt(f"{results_folder}{args.net_basename}_{args.model_type}_pos_cnt.csv", pos_cnt, delimiter=",")
    np.savetxt(f"{results_folder}{args.net_basename}_{args.model_type}_neg_cnt.csv", neg_cnt, delimiter=",")
    np.savetxt(f"{results_folder}{args.net_basename}_{args.model_type}_grad_avg.csv", grad_avg, delimiter=",")
    np.savetxt(f"{results_folder}{args.net_basename}_{args.model_type}_gradsum_avg.csv", gradsum_avg, delimiter=",")
    
    print(f"Model {args.model_type}")
    # Print the stored (averaged across batches) test losses and metrics, per task.
    test_stats = log_test_results(tasks, metric, losses, num_test_batches, aggregators, logger)
    test_stats_corr = log_test_results(tasks, metric_corr, losses_corr, num_test_batches, aggregators_corr, logger)

    # Save test results.
    
    torch.save({'stats': test_stats, 'args': vars(args)},
               f"{results_folder}{args.net_basename}_{args.model_type}_test_results.pkl")
    torch.save({'stats': test_stats_corr, 'args': vars(args)},
               f"{results_folder}{args.net_basename}_{args.model_type}_WA_test_results.pkl")


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
    parser.add_argument('--nih_labels', type=str, default=True, help='NIH labels to be used')
    parser.add_argument('--partial_dataset', type=bool, default=True, help='Use only part of NIH dataset')
    args = parser.parse_args()

    test_multi_task(args, args.random_seed)
