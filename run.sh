# train
python main.py configs/clrnet/clr_dla34_culane.py --gpus 1


# test
python main.py work_dirs/clr/r18_culane/20230113_113829_lr_6e-04_b_24/config.py --validate \
--load_from work_dirs/clr/r18_culane/20230113_113829_lr_6e-04_b_24/ckpt/10.pth \
--gpus 0 \
--view \
