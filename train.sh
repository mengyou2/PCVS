export USE_SLURM=0
export CUDA_VISIBLE_DEVICES=0,1

#train on Tanks and Temples dataset        
#training with gt depth
python train.py --batch-size 1 --folder 'tanksandtemples' --num_workers 1  --model_type zbuffer_pts\
        --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
        --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

#training with mvs depth estimation
# python train.py --batch-size 1 --folder 'tanksandtemples_esdepth' --num_workers 1  --model_type z_buffermodel_w_depth_estimation\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

#training for the refine block
#NEED LOAD CKPT IN PREVIOUS TRAINING
# python train.py --batch-size 1 --folder 'tanksandtemples_hf' --num_workers 1  --model_type z_buffermodel_w_hfrefine\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --reload 'CKPT_PATH'
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

#training for the refine block
#NEED LOAD CKPT IN PREVIOUS TRAINING
# python train.py --batch-size 1 --folder 'tanksandtemples_hf_esdepth' --num_workers 1  --model_type z_buffermodel_w_depth_estimation_w_hfrefine\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --reload 'CKPT_PATH'
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001


#train on DTU dataset
# python train.py --batch-size 1 --folder 'dtu' --num_workers 1  --model_type zbuffer_pts\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

# python train.py --batch-size 1 --folder 'dtu' --num_workers 1  --model_type z_buffermodel_w_depth_estimation\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

# python train.py --batch-size 1 --folder 'dtu' --num_workers 1  --model_type z_buffermodel_w_hfrefine\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --reload 'CKPT_PATH'
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

# python train.py --batch-size 1 --folder 'dtu' --num_workers 1  --model_type z_buffermodel_w_depth_estimation_w_hfrefine\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --reload 'CKPT_PATH'
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

#train on RealEstate10K dataset
# python train.py --batch-size 1 --folder 'RealEstate10K' --num_workers 1  --model_type zbuffer_pts\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

# python train.py --batch-size 1 --folder 'RealEstate10K' --num_workers 1  --model_type z_buffermodel_w_depth_estimation\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

# python train.py --batch-size 1 --folder 'RealEstate10K' --num_workers 1  --model_type z_buffermodel_w_hfrefine\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --reload 'CKPT_PATH'
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001

# python train.py --batch-size 1 --folder 'RealEstate10K' --num_workers 1  --model_type z_buffermodel_w_depth_estimation_w_hfrefine\
#         --resume --dataset 'tanksandtemples' --accumulation 'alphacomposite' --radius 1.5 --depth_radius 3 --gpu_ids 0,1 \
#         --reload 'CKPT_PATH'
#         --k 12 --n_ratio 0.8 --max_epoch 300 --lr 0.00001