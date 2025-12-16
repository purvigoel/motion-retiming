from jackson_import import *
from mdm.utils.fixseed import fixseed
import os
import argparse
import json
import numpy as np
import torch
from gthmr.emp_train.training_loop import TrainLoop
from mdm.utils.parser_util import generate_args, train_args, train_emp_args
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils import dist_util
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from gthmr.emp_train.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
from mdm.utils.model_util import create_emp_model_and_diffusion
import shutil
from mdm.data_loaders.tensors import collate
from gthmr.lib.utils.mdm_utils import viz_motions
from VIBE.lib.dataset.vibe_dataset import rotate_about_D
from gthmr.emp_train.get_data import get_dataset_loader_dict_new2
from gthmr.lib.utils import data_utils
import ipdb
import yaml
from utils.misc import updata_ns_by_missing_keys
from mdm.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_aa, axis_angle_to_6d
import generative_infill.asset_library as asset_library
import generative_infill.dataset_gen_summary_statistics as dataset_gen
from generative_infill.reloader import reload
from generative_infill.to_meshes import to_meshes
#########################
# Load pretrained MR-DM #
#########################

args = train_emp_args()
fixseed(args.seed )

device = 'cuda'

os.makedirs(args.save_dir, exist_ok=True)

def interpolate_loop(to_infill, frames):
    og_motion = to_infill.clone()
    for fr_ind in range(len(frames)):
        if fr_ind == 0:
            prev_frame = 0
        else:
            prev_frame = frames[fr_ind - 1]

        if fr_ind == len(frames) - 1:
            next_frame = 59
        else:
            next_frame = frames[fr_ind + 1]

        frame = frames[fr_ind]

        keyf = to_infill[:, :, :, frame]
        delta = keyf.clone()
        lo = prev_frame
        hi =  next_frame

        for fr in range(lo, frame):
            a = (fr - lo) / (frame - lo)
            og_motion[:, :, :, fr] = og_motion[:, :, :, lo] * (1-a)  + (a) * (delta)

        for fr in range(frame, hi):
            a = (fr - frame) / (hi - frame)
            og_motion[:, :, :, fr] = og_motion[:, :, :, hi] * a  + (1 - a) * (delta)
    to_infill[:, :25*6, : , :] = og_motion[:, :25*6, : , :].clone()
    to_infill[:, 164:, :, :] = og_motion[:, 164:, :, :].clone()
    return to_infill

# # Load and parse data config

args.model_path = "retiming/model000355000.pt"
path_model_args = os.path.join(os.path.dirname(args.model_path), "args.json")
if not os.path.exists(path_model_args):
    raise ValueError(f"Model path [{args.model_path}] must be in the same" \
                    "directory as its model args file: [args.json]")
with open(path_model_args, 'r') as f:
    args_pretrained_model = argparse.Namespace(**json.load(f))

args_pretrained_model.total_batch_size = 5

# Overwrite save_dir
args_pretrained_model.save_dir = args.save_dir

args_pretrained_model.dataset = "amass_hml_keyframe"

# Backward comp
args_pretrained_model = updata_ns_by_missing_keys(args_pretrained_model, args)

# Load Pretrained Model
print("creating model and diffusion...")
model, diffusion = create_emp_model_and_diffusion(args_pretrained_model, None)
model.to(device)
model.rot2xyz.smpl_model.eval()

print(f"Loading checkpoints from [{args.model_path}]...")
state_dict = torch.load(args.model_path)
load_model_wo_clip(model, state_dict)

# generate 100 possible completions
num_samples = 100
max_frames = 60
args.batch_size = num_samples
data = get_dataset_loader(name=args_pretrained_model.dataset,
                                  batch_size=num_samples,
                                  num_frames=max_frames,
                                  data_rep = args_pretrained_model.data_rep,
                                  split='db',
                                  hml_mode='train',
                                  shuffle=False
                                  )

N = num_samples  # batch size
T = max_frames  # this is fixed by the model
video_features = torch.zeros(N, T, 2048)

iterator = iter(data)
sample, model_kwargs = next(iterator)

sample = sample.to(device)

model_kwargs["y"]["keyframe_mask"] = torch.tensor(model_kwargs["y"]["keyframe_mask"]).to(device)
model_kwargs["y"]["features"] = torch.zeros(N, T, 2048)
model_kwargs["y"]["noise_mask"] = model_kwargs["y"]["keyframe_mask"].clone()
model_kwargs["y"]["original"] = sample.clone()
model_kwargs["y"]["action_text"] = np.array(model_kwargs["y"]["action_text"])


#### EXAMPLE MOTION ####
start_motion = torch.tensor(np.load("itr1.npy"))[0].unsqueeze(0)
print(start_motion.shape)
start_motion_clone = start_motion.clone()

# select some keyframes from the ground truth motion and mistime them
start_motion[..., 10] = start_motion[..., 20]
start_motion[..., 20] = start_motion[..., 25]
start_motion[..., 30] = start_motion[..., 38]
start_motion[..., 40] = start_motion[..., 45]
start_motion[..., 50] = start_motion[..., 54]

smoothed_cond = interpolate_loop(start_motion, [10, 20, 30, 40, 50])
smoothed_cond = reload(model, smoothed_cond)
smoothed_cond[:, 150:154, ...] = 0
####

condition_clone = smoothed_cond.clone()
model_kwargs["y"]["keyframe"] = torch.tensor(smoothed_cond).squeeze(2).permute(0, 2, 1).to(sample.device).repeat(num_samples, 1, 1)
smoothed_cond = torch.tensor(smoothed_cond).to(sample.device).repeat(num_samples, 1, 1, 1)


model_kwargs['y']['text'] = [''] * sample.shape[0]
guidance_param = 0.

model.eval()

with torch.no_grad():
    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
                    model,
                    (args.batch_size, model.njoints, model.nfeats, max_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None, #noise,
                    const_noise=False,
                    const_t_noise=False,
                    grad_model=model
    )
    sample = sample.detach()


j_dic = model.forward_kinematics(sample, None)
smpl_joints = j_dic['kp_45_joints'][:, :22].reshape(N, T, 22, 3) + j_dic["pred_trans"].reshape(N, T,3).unsqueeze(-2)

# save out 100 motions in tensor of size 100 x num_frames x num_joints x 3
# you can use tiny-motion-visualizer to display these in a browser window, or your own graphing code
np.save("generated_motion.npy", smpl_joints.cpu().numpy())
