# Description: Evaluate the tracking results of the football dataset using the FieldTrack algorithm

############################################
paths=( 
    #    "/media/hcchen/data/data/football_data/videoData/motions/slowest" 
    #    "/media/hcchen/data/data/football_data/videoData/motions/slow" 
    #    "/media/hcchen/data/data/football_data/videoData/motions/mediumslow"
    #    "/media/hcchen/data/data/football_data/videoData/motions/mediumfast"
    #    "/media/hcchen/data/data/football_data/videoData/motions/fast"
    #    "/media/hcchen/data/data/football_data/videoData/motions/fastest"
         "/media/hcchen/data/data/football_data/videoData/cut2" 
        # "/media/hcchen/data/data/football_data/videoData/nocut0" 
        # "/media/hcchen/data/data/football_data/videoData/nocut1"
        # "/media/hcchen/data/data/football_data/videoData/nocut2"
        )

raw_path="/media/hcchen/data/data/football_data/dataset_.98"
#raw_path="/media/hcchen/data/data/football_data/maincam_momentum"
#raw_path="/media/hcchen/backup/DataCollection/CARLA_0.9.15/output_2"

# Pretrained method
method="MOT17" # MOT17, MOT20, Dance
############################################


# backup config file
cp FieldTrack/config/eval_my.py FieldTrack/config/eval_my.py.backup

# modifying config file
echo "Setting raw path to $raw_path"
sed -i "s|self.raw_video_path = .*|self.raw_video_path = \"$raw_path\"|" FieldTrack/config/eval_my.py

if [ $method == "MOT17" ]; then
    exp_file="exps/example/mot/yolox_x_mix_det.py"
    ckpt="pretrained/ocsort_x_mot17.pth.tar"
elif [ $method == "MOT20" ]; then
    exp_file="exps/example/mot/yolox_x_mix_mot20_ch.py"
    ckpt="pretrained/ocsort_x_mot20.pth.tar"
elif [ $method == "Dance" ]; then
    exp_file="exps/example/mot/yolox_dancetrack_test.py"
    ckpt="pretrained/ocsort_dance_model.pth.tar"
else
    echo "Invalid method"
    exit 1
fi
sed -i "s|self.exp_file = .*|self.exp_file = \"$exp_file\"|" FieldTrack/config/eval_my.py
sed -i "s|self.ckpt = .*|self.ckpt = \"$ckpt\"|" FieldTrack/config/eval_my.py


out_path="results/eval_hist_my_${method}.csv"

# 循环替换路径并运行 Python 程序
for path in "${paths[@]}"; do
    pred_path="football/$(basename $path)"

    echo "Running script with path: $path"

    cd FieldTrack
    sed -i "s|self.path = .*|self.path = \"$path\"|" config/eval_my.py
    # 运行修改后的 Python 程序
    python eval_football_track.py

    cd ..
    python evaluate.py $raw_path $path "FieldTrack/${pred_path}" --output $out_path
done

# 恢复原始的 main.py 文件
mv FieldTrack/config/eval_my.py.backup FieldTrack/config/eval_my.py