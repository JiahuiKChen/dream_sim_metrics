import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"

import torch
from dreamsim import dreamsim
from PIL import Image
import torch.nn.functional as F
from os import listdir
from tqdm import tqdm

model, preprocess = dreamsim(pretrained=True, cache_dir="/datastor1/jiahuikchen/dreamsim_cache")

synth_img_root = "/datastor1/jiahuikchen/synth_ImageNet/"
# synth_img_dirs = [f"{synth_img_root}dropout_90/", f"{synth_img_root}rand_img_cond_90/", 
#     f"{synth_img_root}cutmix_90/", f"{synth_img_root}mixup_90/", f"{synth_img_root}embed_cutmix_90/",
#     f"{synth_img_root}embed_mixup_90/", f"{synth_img_root}embed_cutmix_dropout_90/",
#     f"{synth_img_root}embed_mixup_dropout_90/", f"{synth_img_root}cutmix_dropout_90/", f"{synth_img_root}mixup_dropout_90/"
# ]
synth_img_dirs_0 = [f"{synth_img_root}dropout_90/", f"{synth_img_root}rand_img_cond_90/"] # crashed on MIDI 1 tmux 1
synth_img_dirs_1 = [f"{synth_img_root}cutmix_90/", f"{synth_img_root}mixup_90/"] # MIDI 3 tmux 2
synth_img_dirs_2 = [f"{synth_img_root}embed_cutmix_90/", f"{synth_img_root}embed_mixup_90/"] # A40 tmux 2
synth_img_dirs_3 = [f"{synth_img_root}embed_cutmix_dropout_90/", f"{synth_img_root}embed_mixup_dropout_90/"] # MIDI 3 tmux 1
synth_img_dirs_4 = [f"{synth_img_root}cutmix_dropout_90/", f"{synth_img_root}mixup_dropout_90/"] # A40 tmux 3

real_dirs = ["/data/jiahuic/ImageNetLT_val_test/ImageNet_LT_val_90/", 
    "/data/jiahuic/ImageNetLT_val_test/ImageNet_LT_test_90/"
]


# given a directory of images, generates dreamsim embeddings for all images
# returns list of embeddings 
def gen_all_embeddings(img_dir):
    embeddings = []
    print(f"Creating embeddings for: {img_dir.split('/')[-2]}")

    for file in tqdm(listdir(img_dir)):
        img_path = f"{img_dir}{file}"
        embeddings.append(model.embed(preprocess(Image.open(img_path)).to("cuda")))

    torch.save(embeddings, f"{img_dir.split('/')[-2]}.pt")
    return embeddings


# given a list of embeddings compute average pairwise distance 
# between all possible pairs 
def avg_L2_distance(embeddings):
    dist_total = 0
    pair_count = 0

    for i, embed in tqdm(enumerate(embeddings), total=len(embeddings)):
        for j in range(len(embeddings)):
            if i == j:
                continue
            other_embed = embeddings[j]
            dist = F.pairwise_distance(embed, other_embed)
            dist_total += dist
            pair_count += 1
    
    avg_dist = dist_total / pair_count
    return avg_dist


def calc_L2_pairwise(img_dirs, out_file_name):
    out_str = ""

    for dir in img_dirs:
        dir_name = dir.split('/')[-2]
        embeddings = gen_all_embeddings(dir)
        avg_dist = avg_L2_distance(embeddings)

        dist_str = f"{dir_name}\t{avg_dist}\n"
        print(dist_str)
        out_str += dist_str

    with open(out_file_name, "w") as out_file:
        out_file.write(out_str)


# loads list of embeddings from chkpt_path, calculate and saves avg pariwise L2
def load_embeds_calc_L2(chkpt_path, out_file_name):
    embed_list = torch.load(chkpt_path)
    avg_dist = avg_L2_distance(embed_list)

    name = chkpt_path.split('/')[-1]
    dist_str = f"{name}\t{avg_dist}\n"
    print(dist_str)
    out_str = dist_str

    with open(out_file_name, "w") as out_file:
        out_file.write(out_str)

# TODO: 
# - for each 90 class image gen method: 
#       keep running averages by class (dict) and an overall one
#       calculate similarity between  ALL POSSIBLE PAIRS of synthetic images 
# - for all real images in the 90 subset test set:
#       do the same thing 
# ideally: we see average pairwise distance between synthetic images is higher than real images
# cause then we have introduced "good" diversity (for methods where accuracy goes up notably)

# all synthetic image 90 sets L2 
# calc_L2_pairwise(synth_img_dirs_4, "avg_pairwise_dists_synth_4.txt")

# real validation and test set L2
# calc_L2_pairwise(real_dirs, "avg_pairwise_dists_real.txt")

# MIDI 1 tmux 1 AND A40 TMUX 4 
# load dropout 90 embeddings, calculate L2
load_embeds_calc_L2("/datastor1/jiahuikchen/dreamsim_metrics/dropout_90.pt", "avg_pairwise_dists_dropout_90.txt")
# rand img 90 (crashed and was linked w dropout)
calc_L2_pairwise([f"{synth_img_root}rand_img_cond_90/"] , "avg_pairwise_dists_rand_img_90.txt")