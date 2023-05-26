import argparse
import torch
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from vit import VisionTransformerDiffPruning
from lvvit import LVViTDiffPruning
# build transforms
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--arch', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data-path', default='/dataset/ImageNet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model-path', default=None, help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--base_rate', type=float, default=0.7)

    return parser

def main(args):
    t_resize_crop = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ])

    t_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    base_rate = args.base_rate
    KEEP_RATE = [base_rate, base_rate ** 2, base_rate ** 3]

    if args.arch == 'deit_tiny':
        # PRUNING_LOC = [3,6,9]
        PRUNING_LOC = [4,7,10]
        # PRUNING_LOC = [3,4,5,6,7,8,9]
        KEEP_RATE = [base_rate**(i+1) for i in range(len(PRUNING_LOC))]
        print(f"Creating model: {args.arch}")
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, viz_mode=True
            )
    elif args.arch == 'deit_small':
        PRUNING_LOC = [3,6,9]
        # PRUNING_LOC = [4,7,10]
        print(f"Creating model: {args.arch}")
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, viz_mode=True
            )
    elif args.arch == 'deit_256':
        PRUNING_LOC = [3,6,9] 
        print(f"Creating model: {args.arch}")
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE
            )
    elif args.arch == 'lvvit_s':
        PRUNING_LOC = [4,8,12] 
        print(f"Creating model: {args.arch}")
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE
        )
    elif args.arch == 'lvvit_m':
        PRUNING_LOC = [5,10,15] 
        print(f"Creating model: {args.arch}")
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE
        )
    else:
        raise NotImplementedError

    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print('## model has been successfully loaded')

    image_path = '/home/ssd3/dataset/imagenet/val/n02108551/ILSVRC2012_val_00023737.JPEG'
    image = Image.open(image_path)
    image = t_resize_crop(image)
    im_tensor = t_to_tensor(image).unsqueeze(0)

    device = 'cuda'
    model.to(device)
    model.eval()
    im_tensor = im_tensor.to(device)
    with torch.cuda.amp.autocast():
        output, decisions = model(im_tensor)
    print([decisions[i][0][0].numel() for i in range(len(PRUNING_LOC))])
    decisions = [decisions[i][0][0].cpu().numpy() for i in range(len(PRUNING_LOC))]
    viz = gen_visualization(image, decisions, PRUNING_LOC)

    plt.figure(figsize=(20, 5))
    plt.imshow(viz)
    plt.axis('off')
    plt.savefig('vis.png')


def get_keep_indices(decisions, PRUNING_LOC):
    keep_indices = []
    for i in range(len(PRUNING_LOC)):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices

def gen_masked_tokens(tokens, indices, alpha=0.2):
    indices = [i for i in range(196) if i not in indices]
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens

def recover_image(tokens):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(14, 14, 16, 16, 3).swapaxes(1, 2).reshape(224, 224, 3)
    return image

def gen_visualization(image, decisions, PRUNING_LOC):
    keep_indices = get_keep_indices(decisions, PRUNING_LOC)
    image = np.asarray(image)
    image_tokens = image.reshape(14, 16, 14, 16, 3).swapaxes(1, 2).reshape(196, 16, 16, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, keep_indices[i]))
        for i in range(len(PRUNING_LOC))
    ]
    viz = np.concatenate([image] + stages, axis=1)
    return viz



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)