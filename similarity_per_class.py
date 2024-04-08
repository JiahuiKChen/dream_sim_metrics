from dreamsim import dreamsim
from PIL import Image
import time

model, preprocess = dreamsim(pretrained=True, cache_dir="/datastor1/jiahuikchen/dreamsim_cache")

start = time.time()
img1 = preprocess(Image.open("/datastor1/jiahuikchen/synth_ImageNet/embed_cutmix_90/419_76.jpg")).to("cuda")
img2 = preprocess(Image.open("/datastor1/jiahuikchen/synth_ImageNet/embed_cutmix_90/419_73.jpg")).to("cuda")

distance = model(img1, img2)
end = time.time()
print(f"Time to preproc + get distance between 2 images: {end - start}")

start = time.time()
embedding = model.embed(img1)
end = time.time()
print(f"Time to embed 1 image: {end - start}")

start = time.time()
img1 = preprocess(Image.open("/datastor1/jiahuikchen/synth_ImageNet/embed_cutmix_90/419_76.jpg")).to("cuda")
end = time.time()
print(f"Time to preproc 1 image: {end - start}")

#example use:
# Perceptual similarity metric
# img1 = preprocess(Image.open("img1_path"))
# img2 = preprocess(Image.open("img2_path"))

# distance = model(img1, img2)

# TODO: USE THEIR "FASTER" CODE: https://github.com/ssundaram21/dreamsim?tab=readme-ov-file#image-retrieval 

# TODO: 
# - for each 90 class image gen method: 
#       keep running averages by class (dict) and an overall one
#       calculate similarity between  ALL POSSIBLE PAIRS of synthetic images 
# - for all real images in the 90 subset test set:
#       do the same thing 
# ideally: we see average pairwise distance between synthetic images is higher than real images
# cause then we have introduced "good" diversity (for methods where accuracy goes up notably)
