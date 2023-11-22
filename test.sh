export TANKS='/home/youmeng/data/tanks_and_temples/testing/M60/dense/ibr3d_pw_0.25'
export scene='M60'
export DEBUG=2


python -u eval_tanks.py --old_model ''\
      --result_folder '' --gpu_ids 0 
# python -u eval_dtu.py --old_model ''\
#       --result_folder '' --gpu_ids 0 
# python -u eval_rs10k.py --old_model ''\
#       --result_folder '' --gpu_ids 0 






