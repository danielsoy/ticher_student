import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from einops import rearrange
from AnomalyNet import AnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import load_model
import pickle
import os
from anomaly_detection import get_score_map, visualize

def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to infer on (in data folder)")
    parser.add_argument('--test_size', type=int, default=20, help="Number of batch for the test set")
    parser.add_argument('--n_students', type=int, default=3, help="Number of students network to use")
    parser.add_argument('--patch_size', type=int, default=65, choices=[17, 33, 65])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--visualize', type=bool, default=True, help="Display anomaly map batch per batch")
    parser.add_argument('--calibration_file', type=str, default=None, help="Path to saved calibration parameters")

    # trainer arguments
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    return args

def predict_anomaly(args):
    # Choosing device 
    device = torch.device("cuda:0" if args.gpus else "cpu")
    print(f'Device used: {device}')

    # Teacher network
    teacher = AnomalyNet.create((args.patch_size, args.patch_size))
    teacher.eval().to(device)

    # Load teacher model
    load_model(teacher, f'../model/{args.dataset}/teacher_{args.patch_size}_net.pt')

    # Students networks
    students = [AnomalyNet.create((args.patch_size, args.patch_size)) for _ in range(args.n_students)]
    students = [student.eval().to(device) for student in students]

    # Loading students models
    for i in range(args.n_students):
        model_name = f'../model/{args.dataset}/student_{args.patch_size}_net_{i}.pt'
        load_model(students[i], model_name)

    # Load calibration parameters from file
    calibration_file = args.calibration_file or f'../model/{args.dataset}/calibration_{args.patch_size}.pkl'
    if os.path.exists(calibration_file):
        print(f"Loading calibration parameters from {calibration_file}")
        with open(calibration_file, 'rb') as f:
            params = pickle.load(f)
    else:
        print(f"Calibration file {calibration_file} not found. Please run calibration first.")
        return

    # Load testing data
    test_dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                                  transform=transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                  gt_transform=transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor()]),
                                  type='test')

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=True, 
                                 num_workers=args.num_workers)

    # Build anomaly map
    y_score = np.array([])
    y_true = np.array([])
    test_iter = iter(test_dataloader)

    for i in range(min(args.test_size, len(test_dataset))):
        batch = next(test_iter)
        inputs = batch['image'].to(device)
        gt = batch['gt'].cpu()

        score_map = get_score_map(inputs, teacher, students, params).cpu()
        y_score = np.concatenate((y_score, rearrange(score_map, 'b h w -> (b h w)').numpy()))
        y_true = np.concatenate((y_true, rearrange(gt, 'b c h w -> (b c h w)').numpy()))

        if args.visualize:
            unorm = transforms.Normalize((-1, -1, -1), (2, 2, 2)) # get back to original image
            max_score = (params['students']['err']['max'] - params['students']['err']['mu']) / torch.sqrt(params['students']['err']['var'])\
                + (params['students']['var']['max'] - params['students']['var']['mu']) / torch.sqrt(params['students']['var']['var']).item()
            img_in = rearrange(unorm(inputs).cpu(), 'b c h w -> b h w c')
            gt_in = rearrange(gt, 'b c h w -> b h w c')

            for b in range(args.batch_size):
                visualize(img_in[b, :, :, :].squeeze(), 
                          gt_in[b, :, :, :].squeeze(), 
                          score_map[b, :, :].squeeze(), 
                          max_score)
    
    # AUC ROC muestra curva de entrenamiento.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_score)
    plt.figure(figsize=(13, 3))
    plt.plot(fpr, tpr, 'r', label="ROC")
    plt.plot(fpr, fpr, 'b', label="random")
    plt.title(f'ROC AUC: {auc(fpr, tpr)}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.show()
    """

if __name__ == '__main__':
    args = parse_arguments()
    predict_anomaly(args)