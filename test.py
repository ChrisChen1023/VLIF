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
from thop import profile, clever_format

# 添加功耗计算器路径
sys.path.append('/home3/shpb49/Postdoc/VLIFNet_all_ablation')
from energy_consumption_calculator import EnergyCalculator

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


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_flops(model, input_size=(1, 3, 256, 256)):
    """计算模型FLOPS"""
    # 创建一个随机输入
    input_tensor = torch.randn(input_size)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    # 使用thop计算FLOPS
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params

def print_model_info(model, input_size=(1, 3, 256, 256), crop_size=80):
    """打印模型信息"""
    print("=" * 80)
    print("                          模型信息")
    print("=" * 80)
    
    # 1. 参数数量
    total_params, trainable_params = count_parameters(model)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
    
    # 2. FLOPS计算 (使用裁剪块的尺寸)
    try:
        crop_input_size = (input_size[0], input_size[1], crop_size, crop_size)
        flops, _ = calculate_flops(model, crop_input_size)
        flops_str, params_str = clever_format([flops, total_params], "%.3f")
        print(f"FLOPS (per {crop_size}x{crop_size} patch): {flops_str}")
        print(f"参数数量 (thop): {params_str}")
        
        # 重置模型状态，避免尺寸不匹配问题
        functional.reset_net(model)
    except Exception as e:
        print(f"FLOPS计算失败: {e}")
        flops = 0
    
    # 3. 能耗计算
    print("\n" + "=" * 80)
    print("                          能耗分析")
    print("=" * 80)
    
    try:
        # 创建能耗计算器
        calculator = EnergyCalculator(T=4, sparsity=0.1642)
        
        # Calculate energy consumption (Use full image dimensions)
        C, H, W = input_size[1], input_size[2], input_size[3]
        total_energy = calculator.calculate_vlifnet_energy(
            input_size=(C, H, W),
            dim=24,  # Adjust based on actual model
            en_num_blocks=[4, 4, 6, 6],
            de_num_blocks=[4, 4, 6, 6]
        )
        
        print(f"\n能耗汇总 (完整 {H}x{W} 图像):")
        print(f"单张图像处理能耗: {total_energy*1e12:.2f} pJ")
        print(f"单张图像处理能耗: {total_energy*1e9:.2f} nJ")
        print(f"单张图像处理能耗: {total_energy*1e6:.2f} μJ")
        
        # 计算裁剪块的能耗
        crop_ratio = (crop_size / H) * (crop_size / W)
        crop_energy = total_energy * crop_ratio
        print(f"\n单个 {crop_size}x{crop_size} 裁剪块能耗:")
        print(f"裁剪块处理能耗: {crop_energy*1e12:.2f} pJ")
        print(f"裁剪块处理能耗: {crop_energy*1e9:.2f} nJ")
        
        # 计算每秒能耗 (假设处理10 FPS)
        fps = 10
        power_consumption = total_energy * fps
        print(f"\n假设处理速度 {fps} FPS:")
        print(f"功耗: {power_consumption*1e6:.2f} μW")
        print(f"功耗: {power_consumption*1e3:.2f} mW")
        
    except Exception as e:
        print(f"能耗计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)


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
    checkpoint = torch.load(weights)
    try:
        if "state_dict" in checkpoint:
            print("Loading from checkpoint with 'state_dict' key...")
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("Loading directly from checkpoint...")
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"First attempt failed: {e}")
        print("Trying to remove 'module.' prefix...")
        try:
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
                
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.` if exists
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception as e2:
            print(f"Second attempt also failed: {e2}")
            print("Trying to load with strict=False...")
            try:
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                    
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=False)
                print("Successfully loaded with strict=False (some keys may be missing)")
            except Exception as e3:
                print(f"All attempts failed: {e3}")
                raise e3


if __name__ == '__main__':
    model_restoration = model.cuda()
    functional.set_step_mode(model_restoration, step_mode='m')
    functional.set_backend(model_restoration, backend='cupy')
    
    # 正确加载checkpoint
    print(f"Loading model weights from: {opt.weights}")
    load_checkpoint(model_restoration, opt.weights)
    
    print("===>Testing using weights: ", opt.weights)
    model_restoration.cuda()
    model_restoration.eval()
    
    # 打印模型信息
    print_model_info(model_restoration, input_size=(1, 3, 256, 256), crop_size=crop_size)
    
    inp_dir = opt.data_path
    eval_loader = getevalloader(opt)
    result_dir = opt.save_path
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize metrics
    psnr_list = []
    ssim_list = []
    
    # 开始测试
    print("\n" + "=" * 80)
    print("                          开始测试")
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
    
    # 测试完成统计
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
    print(f"总测试时间: {test_time:.2f} 秒")
    print(f"平均每张图像处理时间: {test_time/num_images:.2f} 秒")
    print(f"实际处理速度: {num_images/test_time:.2f} FPS")
    
    # Save results to file
    results_file = os.path.join(result_dir, 'metrics_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Total images processed: {len(psnr_list)}\n")
        f.write(f"总测试时间: {test_time:.2f} 秒\n")
        f.write(f"平均每张图像处理时间: {test_time/num_images:.2f} 秒\n")
        f.write(f"实际处理速度: {num_images/test_time:.2f} FPS\n\n")
        f.write("Per-image results:\n")
        for i, (psnr, ssim) in enumerate(zip(psnr_list, ssim_list)):
            f.write(f"Image {i+1}: PSNR={psnr:.4f}, SSIM={ssim:.4f}\n")
    
    print(f"Results saved to: {results_file}")
