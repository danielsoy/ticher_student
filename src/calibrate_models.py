import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import os
from AnomalyNet import AnomalyNet
from AnomalyDataset import AnomalyDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from utils import load_model
from anomaly_detection import calibrate

def parse_arguments():
    parser = ArgumentParser()

    # program arguments
    parser.add_argument('--dataset', type=str, default='carpet', help="Dataset to calibrate on")
    parser.add_argument('--n_students', type=int, default=3, help="Number of students network to use")
    parser.add_argument('--patch_size', type=int, default=65, choices=[17, 33, 65])
    parser.add_argument('--image_size', type=int, default=256)

    # trainer arguments
    parser.add_argument('--gpus', type=int, default=(1 if torch.cuda.is_available() else 0))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    return args

def calibrate_models(args):
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

    # calibration on anomaly-free dataset
    calib_dataset = AnomalyDataset(root_dir=f'../data/{args.dataset}',
                                    transform=transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    type='train',
                                    label=0)

    calib_dataloader = DataLoader(calib_dataset, 
                                   batch_size=args.batch_size, 
                                   shuffle=False, 
                                   num_workers=args.num_workers)
    
    print("Starting calibration process...")
    params = calibrate(teacher, students, calib_dataloader, device)
    
    # Save calibration parameters
    os.makedirs(f'../model/{args.dataset}', exist_ok=True)
    calibration_file = f'../model/{args.dataset}/calibration_{args.patch_size}.pkl'
    with open(calibration_file, 'wb') as f:
        pickle.dump(params, f)
    
    print(f"Calibration parameters saved to {calibration_file}")

if __name__ == '__main__':
    args = parse_arguments()
    calibrate_models(args)