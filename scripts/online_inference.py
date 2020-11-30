

from argparse import Namespace
import time
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

sys.path.append(".")
sys.path.append("..")

from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp

import dlib
from scripts.align_all_parallel import align_face
# %load_ext autoreload
# %autoreload 2

# Step 1: Select Experiment Type
# experiment_type = 'ffhq_encode'
# experiment_type = 'ffhq_frontalize'
# experiment_type = 'celebs_sketch_to_face'
# experiment_type = 'celebs_seg_to_face'
# experiment_type = 'celebs_super_resolution'
experiment_type = 'toonify'

# Step 2: Download Pretrained Models
# download petrained model
#def get_download_model_command(file_id, file_name):
#    """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
#    current_directory = os.getcwd()
#    save_path = os.path.join(os.path.dirname(current_directory), CODE_DIR, "pretrained_models")
#    if not os.path.exists(save_path):
#        os.makedirs(save_path)
#    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
#    return url

MODEL_PATHS = {
    "ffhq_encode": {"id": "1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "name": "psp_ffhq_encode.pt"},
    "ffhq_frontalize": {"id": "1_S4THAzXb-97DbpXmanjHtXRyKxqjARv", "name": "psp_ffhq_frontalization.pt"},
    "celebs_sketch_to_face": {"id": "1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA", "name": "psp_celebs_sketch_to_face.pt"},
    "celebs_seg_to_face": {"id": "1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz", "name": "psp_celebs_seg_to_face.pt"},
    "celebs_super_resolution": {"id": "1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu", "name": "psp_celebs_super_resolution.pt"},
    "toonify": {"id": "1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz", "name": "psp_ffhq_toonify.pt"}
}

# path = MODEL_PATHS[experiment_type]
# download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
# wget {download_command}

# Step 3: Define Inference Parameters
EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pretrained_models/psp_ffhq_encode.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "ffhq_frontalize": {
        "model_path": "pretrained_models/psp_ffhq_frontalization.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celebs_sketch_to_face": {
        "model_path": "pretrained_models/psp_celebs_sketch_to_face.pt",
        "image_path": "notebooks/images/input_sketch.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    },
    "celebs_seg_to_face": {
        "model_path": "pretrained_models/psp_celebs_seg_to_face.pt",
        "image_path": "notebooks/images/input_mask.png",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.ToOneHot(n_classes=19),
            transforms.ToTensor()])
    },
    "celebs_super_resolution": {
        "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
         "model_path": "pretrained_models/psp_ffhq_toonify.pt",
         #"image_path": "notebooks/images/input_img.jpg",
        #  "model_path": '/content/drive/MyDrive/Colab Notebooks/PsP/pixel2style2pixel'+"/"+"pretrained_models/psp_ffhq_toonify.pt",
        "image_path": '/content/drive/MyDrive/Colab Notebooks/PsP/pixel2style2pixel'+"/"+"notebooks/images/image_b_256256.jpg",
         "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

# verify that the model
print("EXPERIMENT_ARGS['model_path']:",EXPERIMENT_ARGS['model_path'])
if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
  raise ValueError("Pretrained model was unable to be downlaoded correctly!")

## Step 4: Load Pretrained Model
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')


opts = ckpt['opts']
pprint.pprint(opts)

# update the training options
opts['checkpoint_path'] = model_path
if 'learn_in_w' not in opts:
    opts['learn_in_w'] = False

opts= Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

## Step 5: Visualize Input

# Alignment face
def run_alignment(image_path):
  predictor = dlib.shape_predictor("./pretrained_models/shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image


def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


def online_inference_fn(image_path,output_path):
    # Step1: 步骤1，加载图片，图片大小调整。load image and resize
    # resize image first
    # Visiual input
    # image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
    file_name=image_path.split("/")[-1]
    original_image = Image.open(image_path)
    if opts.label_nc == 0:
        original_image = original_image.convert("RGB")
    else:
        original_image = original_image.convert("L")
    #original_image.resize((256, 256)) # resize
    original_image = original_image.resize((256, 256), resample=Image.BILINEAR)

    # Step2: 步骤2： 裁剪出人脸。run_alignment
    if experiment_type not in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
        try:
            input_image = run_alignment(image_path)
            print("run alignment error!...")
        except Exception as e:
            input_image = original_image
    else:
        input_image = original_image

    # Step 3: 运行生成程序 Perform Inference

    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)

    if experiment_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
        latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    else:
        latent_mask = None
    with torch.no_grad():
        tic = time.time()
        result_image = run_on_batch(transformed_image.unsqueeze(0), net, latent_mask)[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    # Step 6.2: Visualize Result
    input_vis_image = log_input_image(transformed_image, opts)
    output_image = tensor2im(result_image)
    if experiment_type == "celebs_super_resolution":
        res = np.concatenate([np.array(input_image.resize((256, 256))),
                              np.array(input_vis_image.resize((256, 256))),
                              np.array(output_image.resize((256, 256)))], axis=1)
    else:
        res = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                              np.array(output_image.resize((256, 256)))], axis=1)
    res_image = Image.fromarray(res) # 实现array到image的转换, return an image object

    # res_image
    # save image to output path
    res_image.save(output_path+"_"+file_name+"_toon.jpg")
image_path='./resources/images/input_img_2.jpg'
output_path='./resources/images_output/'
online_inference_fn(image_path,output_path)