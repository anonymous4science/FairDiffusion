# FairDiffusion

Despite the strong performance of these generative models, it remains an open question whether the quality of image generation is consistent across different demographic subgroups. To tackle these biases, we introduce FairDiffusion, an equity-aware latent diffusion model that enhances fairness in medical image generation via Fair Bayesian Perturbation.

## Requirements

To install the prerequisites, run:
```
conda env create -f environment.yml
```
or
```
pip install -r requirements.txt
```

## FairGenMed Dataset

We present FairGenMed, the first dataset for studying fairness of medical generative models, providing detailed quantitative measurements of multiple clinical conditions to investigate the semantic correlation between text prompts and anatomical regions across various demographic subgroups.

The dataset can be accessed via this [link](https://drive.google.com/drive/folders/16VCgkvMdCM7UFRlhTlEa0oV2lxCho6Al?usp=sharing). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

Our dataset includes 10,000 subjects for glaucoma detection with comprehensive demographic identity attributes including age, gender, race, ethnicity, preferred language, and marital status. Each subject has one Scanning Laser Ophthalmoscopy (SLO) fundus photo and one npz file. The size of SLO fundus photos is 512 x 664.

The NPZ files have the following attributes
```
glaucoma: the label of glaucoma disease, 0 - non-glaucoma, 1 - glaucoma
oct_bscans: images of OCT B-scans
race: 0 - Asian, 1 - Black, 2 - White
male: 0 - Female, 1 - Male
hispanic: 0 - Non-Hispanic, 1 - Hispanic
maritalstatus: 0 - Married or Partnered, 1 - Single, 2 - Divorced, 3 - Widowed, 4 - Legally Separated, and -1 - Unknown
language: 0 - English, 1 - Spanish, 2 - Others
```

We have compiled all clinical measurements related to the 10,000 samples into a meta CSV file named *data_summary.csv*. Specifically, the cup-disc ratio, severity of vision loss, and status of spherical equivalent are denoted by the column names 'cdr_status', 'md_severity', and 'se_status', respectively, in the data_summary.csv file.


## Experiments
**Train Stable Diffusion**
```
cd examples/text_to_image
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
accelerate launch --main_process_port 29601 --multi_gpu --num_processes 2 --mixed_precision="fp16" train_text_to_image.py --text_encoder_type clip --dataset_dir <DATASET_DIR> --checkpointing_steps 5000 --pretrained_model_name_or_path=$MODEL_NAME --train_data_dir tmpp --datasets glaucoma --race_prompt --gender_prompt --ethnicity_prompt --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=16 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=100000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir=<OUTPUT_DIR> --cache_dir <CACHE_DIR>
```

**Train FairDiffusion**
```
cd examples/text_to_image
export MODEL_NAME="stabilityai/fair-diffusion-2-1"
export TIME_WINDOW=30
export EXPLOITATION=0.95
accelerate launch --main_process_port 29601 --multi_gpu --num_processes 2 --mixed_precision="fp16" train_text_to_image.py --text_encoder_type clip --dataset_dir <DATASET_DIR> --checkpointing_steps 5000 --pretrained_model_name_or_path=$MODEL_NAME --train_data_dir tmpp --datasets glaucoma --race_prompt --gender_prompt --ethnicity_prompt --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=16 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=100000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir=<OUTPUT_DIR> --cache_dir <CACHE_DIR> --fair_time_window ${TIME_WINDOW} --fair_exploitation_rate ${EXPLOITATION}
```

**Visualize Diffusion Model (Generation)**
```
python visualize_fairdiffusion.py --dataset_dir <DATASET_DIR> --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --datasets glaucoma --race_prompt --gender_prompt --ethnicity_prompt --vis_dir <VIS_DIR> --prompts prompts.txt --repeat_prompt 5
```

**Evaluate Fairness of Diffusion Model (Generation)**
```
python evaluate_fairdiffusion.py --metrics_calculation_idx 1 --dataset_dir <DATASET_DIR> --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --datasets glaucoma --race_prompt --gender_prompt --ethnicity_prompt > OUTPUT.txt
```

**Evaluate Fairness of Diffusion Model (Classification) - ViT-B**
```
cd classification_codebase
DATASET_DIR=
RESULT_DIR=
MODEL_TYPE=ViT-B
MODALITY_TYPE=slo_fundus

VIT_WEIGHTS=imagenet
BATCH_SIZE=64
BLR=5e-4
WD=0.01
LD=0.55
DP=0.1

# Baselines
EXP_NAME=BASELINE_GLAUCOMA
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cls --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

EXP_NAME=BASELINE_MD_SEVERITY
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task md_severity --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

EXP_NAME=BASELINE_CDR_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cdr_status --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

EXP_NAME=BASELINE_SE_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task se_status --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

# Generated Dataset
EXP_NAME=FAIRDIFFUSION_GLAUCOMA
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cls --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

EXP_NAME=FAIRDIFFUSION_MD_SEVERITY
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task md_severity --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

EXP_NAME=FAIRDIFFUSION_CDR_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cdr_status --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

EXP_NAME=FAIRDIFFUSION_SE_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task se_status --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --epochs 50 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}

```


**Evaluate Fairness of Diffusion Model (Classification) - EfficientNet**
```
cd classification_codebase

DATASET_DIR=
RESULT_DIR=
MODEL_TYPE=( efficientnet )
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus'
ATTRIBUTE_TYPE=race

OPTIMIZER='adamw'
OPTIMIZER_ARGUMENTS='{"lr": 0.001, "weight_decay": 0.01}'

SCHEDULER='step_lr'
SCHEDULER_ARGUMENTS='{"step_size": 30, "gamma": 0.1}'

LR=1e-3

# Baselines
BATCH_SIZE=6
EXP_NAME=BASELINE_GLAUCOMA
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cls --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

BATCH_SIZE=6
EXP_NAME=BASELINE_MD_SEVERITY
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task md_severity --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

BATCH_SIZE=16
EXP_NAME=BASELINE_CDR_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cdr_status --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

BATCH_SIZE=16
EXP_NAME=BASELINE_SE_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task se_status --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

# Generated Dataset
BATCH_SIZE=6
EXP_NAME=FAIRDIFFUSION_GLAUCOMA
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cls --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

BATCH_SIZE=6
EXP_NAME=FAIRDIFFUSION_MD_SEVERITY
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task md_severity --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

BATCH_SIZE=16
EXP_NAME=FAIRDIFFUSION_CDR_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task cdr_status --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

BATCH_SIZE=16
EXP_NAME=FAIRDIFFUSION_SE_STATUS
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
python scripts/train_glaucoma_fair.py --task se_status --fairdiffusion --initial_model stabilityai/stable-diffusion-2-1 --model_path <OUTPUT_DIR>/checkpoint-<TBD>/unet --data_dir ${DATASET_DIR}/ --result_dir ${RESULT_DIR}/${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --image_size 200 --lr ${LR} --weight-decay 0. --momentum 0.1 --batch_size ${BATCH_SIZE} --epochs ${NUM_EPOCH} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE}

```