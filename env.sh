# create env
conda create -n au_CLRNet python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# dataset
# 用到了laneseg_lable_w16文件夹，6个driver开头的frame文件夹，以及list文件夹
ln -s /media/peter/ocean/data/dataset/lane/CULane data/CULane
