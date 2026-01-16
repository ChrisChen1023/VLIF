import torch
import os
# from utils.common import print_network
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import utils
import argparse
import math
from tqdm import tqdm
from model import model
import torch.utils.data as data
from glob import glob
from PIL import Image
from spikingjelly.activation_based import functional
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import sys
import time
from thop import profile


parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', type=str, default='crop')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. -1 for CPU')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--target_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--eval_workers', type=int, default=4)
parser.add_argument('--crop_size', type=int, default=80)
parser.add_argument('--overlap_size', type=int, default=8)
parser.add_argument('--weights', type=str, default='')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

crop_size = opt.crop_size
overlap_size = opt.overlap_size
batch_size = opt.batch_size


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Check image dimensions and adjust window size if needed
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim) if min_dim < 7 else 7
    # Ensure win_size is odd
    if win_size % 2 == 0:
        win_size -= 1
    
    return structural_similarity(img1, img2, 
                               channel_axis=2,  # Use channel_axis instead of multichannel
                               data_range=1.0,
                               win_size=win_size)


class DataLoaderEval(data.Dataset):
    def __init__(self, opt):
        super(DataLoaderEval, self).__init__()
        self.opt = opt
        # Get input images
        inp_imgs = glob(os.path.join(opt.data_path, '*.png')) + glob(os.path.join(opt.data_path, '*.jpg'))
        
        # Get target images
        tar_imgs = glob(os.path.join(opt.target_path, '*.png')) + glob(os.path.join(opt.target_path, '*.jpg'))

        if len(inp_imgs) == 0:
            raise (RuntimeError("Found 0 input images in: " + opt.data_path + "\n"))
        if len(tar_imgs) == 0:
            raise (RuntimeError("Found 0 target images in: " + opt.target_path + "\n"))
        
        # Sort to ensure matching pairs
        inp_imgs.sort()
        tar_imgs.sort()
        
        # Create pairs based on filename matching
        self.img_pairs = []
        for inp_path in inp_imgs:
            inp_name = os.path.basename(inp_path)
            # Find corresponding target image
            tar_path = None
            for tar_p in tar_imgs:
                if os.path.basename(tar_p) == inp_name:
                    tar_path = tar_p
                    break
            
            if tar_path is not None:
                self.img_pairs.append((inp_path, tar_path))
            else:
                print(f"Warning: No target image found for {inp_name}")
        
        self.sizex = len(self.img_pairs)
        print(f"Found {self.sizex} matching image pairs")

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path, tar_path = self.img_pairs[index_]
        
        # Load input image
        inp_img = Image.open(inp_path).convert('RGB')
        inp_img = TF.to_tensor(inp_img)
        
        # Load target image
        tar_img = Image.open(tar_path).convert('RGB')
        tar_img = TF.to_tensor(tar_img)
        
        return inp_img, tar_img, os.path.basename(inp_path)


def getevalloader(opt):
    dataset = DataLoaderEval(opt)
    print("Dataset Size:%d" % (len(dataset)))
    evalloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.eval_workers,
                                 pin_memory=True)
    return evalloader


def splitimage(imgtensor, crop_size=crop_size, overlap_size=overlap_size):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=batch_size, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size=crop_size, resolution=(batch_size, 3, crop_size, crop_size)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


from collections import OrderedDict


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)

    if isinstance(state_dict, dict):
        state_dict = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items()
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print(f"Loaded weights (strict=False). Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


if __name__ == '__main__':
    model_restoration = model.cuda()
    functional.set_step_mode(model_restoration, step_mode='m')
    functional.set_backend(model_restoration, backend='cupy')
    
    # Correctly load checkpoint
    print(f"Loading model weights from: {opt.weights}")
    load_checkpoint(model_restoration, opt.weights)
    
    print("===>Testing using weights: ", opt.weights)
    model_restoration.cuda()
    model_restoration.eval()
    
    # FLOPs / Params
    x = torch.rand(1, 3, 256, 256).cuda()
    functional.set_step_mode(model_restoration, step_mode='m')
    functional.set_backend(model_restoration, backend='cupy')
    flops, params = profile(model_restoration, inputs=(x,), verbose=False)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    functional.reset_net(model_restoration)
    
    inp_dir = opt.data_path
    eval_loader = getevalloader(opt)
    result_dir = opt.save_path
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize metrics
    psnr_list = []
    ssim_list = []
    
    # Start testing
    print("\n" + "=" * 80)
    print("                          Start Testing")
    print("=" * 80)
    
    start_time = time.time()
    with torch.no_grad():
        for input_, target_, file_ in tqdm(eval_loader, unit='img'):
            input_ = input_.cuda()
            target_ = target_.cuda()
            B, C, H, W = input_.shape
            
            # Process input image
            split_data, starts = splitimage(input_)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cuda()
                functional.reset_net(model_restoration)
                split_data[i] = split_data[i].cpu()

            restored = mergeimage(split_data, starts, resolution=(B, C, H, W))
            restored = torch.clamp(restored, 0, 1)
            
            # Calculate metrics for each image in batch
            for j in range(B):
                fname = file_[j]
                
                # Convert to numpy for saving and metrics calculation
                restored_np = restored[j].permute(1, 2, 0).numpy()
                target_np = target_[j].permute(1, 2, 0).cpu().numpy()
                
                # Calculate PSNR and SSIM
                psnr_val = calculate_psnr(target_np, restored_np)
                ssim_val = calculate_ssim(target_np, restored_np)
                
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                
                print(f"Image: {fname}, PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
                
                # Save restored image
                cleanname = fname
                save_file = os.path.join(result_dir, cleanname)
                save_img(save_file, img_as_ubyte(restored_np))
    
    # Testing statistics
    end_time = time.time()
    test_time = end_time - start_time
    num_images = len(psnr_list)
    
    # Calculate and print average metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print(f"\n=== Final Results ===")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Total images processed: {len(psnr_list)}")
    print(f"Total test time: {test_time:.2f} seconds")
    print(f"Average time per image: {test_time/num_images:.2f} seconds")
    print(f"Actual processing speed: {num_images/test_time:.2f} FPS")
    
    # Save results to file
    results_file = os.path.join(result_dir, 'metrics_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Total images processed: {len(psnr_list)}\n")
        f.write(f"Total test time: {test_time:.2f} seconds\n")
        f.write(f"Average time per image: {test_time/num_images:.2f} seconds\n")
        f.write(f"Actual processing speed: {num_images/test_time:.2f} FPS\n\n")
        f.write("Per-image results:\n")
        for i, (psnr, ssim) in enumerate(zip(psnr_list, ssim_list)):
            f.write(f"Image {i+1}: PSNR={psnr:.4f}, SSIM={ssim:.4f}\n")
    
    print(f"Results saved to: {results_file}")
