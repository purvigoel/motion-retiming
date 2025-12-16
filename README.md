## Generative Motion Infilling from Imprecisely Timed Keyframes  ##
(Eurographics 2025)
Purvi Goel, Haotian Zhang, C. Karen Liu, Kayvon Fatahalian

Keyframes are a standard representation for kinematic motion specification. Recent learned motion-inbetweening methods use keyframes as a way to control generative motion models, and are trained to generate life-like motion that matches the exact poses and timings of input keyframes. However, the quality of generated motion may degrade if the timing of these constraints is not perfectly consistent with the desired motion. Unfortunately, correctly specifying keyframe timings is a tedious and challenging task in practice.  Our goal is to create a system that synthesizes high-quality motion from keyframes, even if keyframes are imprecisely timed. We present a method that allows constraints to be retimed as part of the generation process. Specifically, we introduce a novel model architecture that explicitly outputs a time-warping function to correct mistimed keyframes, and spatial residuals that add pose details.  We demonstrate how our method can automatically turn approximately timed keyframe constraints into diverse, realistic motions with plausible timing and detailed submovements.

## Intro ##

This is the official code release for "Generative Motion Infilling from Imprecisely Timed Keyframes". Much of this codebase is shared from our earlier animation project, "Iterative Editing with Natural Language", and many of the set-up instructions are thus the same.

The project was implemented within a much larger codebase, built by <a href="https://wangkua1.github.io/">Kuan-Chieh Wang</a> and <a href="https://jmhb0.github.io/">James Burgess</a>, which itself was built off of great works including <a href=https://github.com/GuyTevet/motion-diffusion-model>MDM</a>, <a href="https://github.com/mkocabas/VIBE">VIBE</a>, <a href="https://github.com/akanazawa/hmr">HMR</a>, and <a href="https://smpl.is.tue.mpg.de/">SMPL</a>.
Most of the relevant files are `generative_infill/retiming.py` (entry point for inference), `mdm/model/mdm.py` (where learned retiming module lives), and `generative_infill/change_time/timewarp.py` (implementation of timewarp using `torch.nn.functional.grid_sample`).

## Getting Started ##

#### Setting up the full system -- here we go! ####

1. Clone the repo.

2. Set up the conda environment snapshotted in `environment.yml` with the command `conda env create -f environment.yml`

3. Download dummy data. This data is important for setting up dataloaders/model inputs. It consists of some motions from <a href="https://amass.is.tue.mpg.de/">AMASS</a>, e.g., ACCAD. Please follow the rules specified in the AMASS website for using their dataset, including citing their work appropriately if you use it in research publications. Ensure that you have the necessary permissions and adhere to their terms of use.
   <br>
   a) The dummy data can be downloaded at this Google Drive link: https://drive.google.com/file/d/1ju-aeHJ8hNBJrmDwG9FslA4f53Q4l5S_/view?usp=sharing
   <br>
   b) Place the data at the path `VIBE/data/vibe_db/amass_db_small.pt`

5. Download the SMPL body model. Specifically, we use `SMPL_NEUTRAL.pkl`.
   <br>
   a) You can download the model from the website https://smpl.is.tue.mpg.de/. You will have to create an account and sign in.
   <br>
   b) Place SMPL_NEUTRAL.pkl at the path `body_models/smpl/SMPL_NEUTRAL.pkl`

7. Download diffusion model. We used the provided model to create all qualitative examples; it was trained on a subset of AMASS data.
   <br>
   a) You can download the models at this Google Drive link: [https://drive.google.com/drive/folders/1ZcKEssmmLopnoh2NDDpX9xoSRxYdKzcJ?usp=drive_link](https://drive.google.com/drive/folders/1JFrwNrZnu7ib93sLljkPGINjUl8APXPJ?usp=sharing).
   <br>
   b) Place the folder in the root directory, i.e., `motion_retiming/retiming/*`
https://github.com/purvigoel/motion-retiming/blob/retiming/generative_infill/retiming.py#L134
That's all the downloads complete! Now we can move on to actually running the model. To run inference, we'll be using `generative_infill/retiming.py`. Have a look at where the <a href="https://github.com/purvigoel/motion-retiming/blob/retiming/generative_infill/retiming.py#L134">source motion is loaded</a>. We provide 3 example motions that can be used as source motions in the `asset_library/` folder. You can see what the motions look like via the videos included in the `asset_library/` folder.

If you'd like to use your own motions, you'll have to convert them into our data representation. <b> Note that our data representation is different from the popular HumanML3D representation. </b> I'm including a note at the end of this section with some pointers about how to do the conversion.

1. Set the environment variable `BIO_POSE_ROOT` to the path to the repo, e.g., `export BIO_POSE_ROOT=/full/path/to/motion-retiming/`

2. You can run the system using the following command: `CUDA_VISIBLE_DEVICES=0 bash run_scripts/ghmr/retiming.sh 0`. You'll see a lot of print-outs, and a tqdm bar as the diffusion model runs.

3. The output motion will be saved as `generated_motion.npy`.
   <br>
   a) The motion is written out as SMPL joints, into a numpy file. The data has shape (batch_size, 60, 22, 3): 60 frames, 22 joints, 3 XYZ world-space positions per joint.
   <br>
   b) You can use your own tools to visualize these motions (I believe the original <a href="https://github.com/GuyTevet/motion-diffusion-model">MDM repository</a> comes with one). I've also written a no-frills <a href="https://github.com/purvigoel/tiny-motion-visualizer.git">web-based skeleton visualizer</a> that handles this data and visualizes it at a localhost port. Run it with `python3 viewer.py -p <PORT> -d <PATH_TO_JOINTS_FOLDER>

#### Motion Representation ####

If you'd like to use your own SMPL source motion instead of the ones provided in our `asset_library/`, you'll need to make sure the motion is converted into our data representation. Our data representation has shape (Batch size, 236, 1, 60). The feature dimension is 236, and we handle 60 frame motions.

The 236-dimensional feature vector comprises the following: 24 SMPL joint rotations converted to rotation6d representation and flattened (24*6 = 144), 3D world space translation of the root followed by a 3 zero paddings (3 + 3 = 6). Then we include a 4-dim foot contact label (4) and a 10-dim SMPL body shape parameter (which we just set to 0). Finally, we include the 3D world-space positions of the joints (24 * 3 = 72) as a redundant representation.

Overall, 24*6 + 3 + 3 + 4 + 10 + 24 * 3 = 236.

Check out the <a href="https://github.com/purvigoel/iterative-editing-release/blob/new-branch4/generative_infill/reloader.py#L26">reloader.py</a> for some code on calculating the world space positions and foot contact labels, once you've got your joint rotations.

### Development ###

This is a research prototype and, in general, will not be regularly maintained long after release. 

### Acknowlegements ###
This work was supported by a Stanford Interdisciplinary Graduate Fellowship.
