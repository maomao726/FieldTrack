# Description: 用于运行 OC_SORT 的评估脚本

############################################
paths=(
    # "/media/hcchen/backup/DataCollection/CARLA_0.9.15/carla_combined/cut"
    # "/media/hcchen/backup/DataCollection/CARLA_0.9.15/carla_combined/slowest"
    # "/media/hcchen/backup/DataCollection/CARLA_0.9.15/carla_combined/slow"
    # "/media/hcchen/backup/DataCollection/CARLA_0.9.15/carla_combined/fast"
    # "/media/hcchen/backup/DataCollection/CARLA_0.9.15/carla_combined/fastest"
     "/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined/cut"
    # "/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined/slowest"
    # "/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined/slow"
    # "/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined/fast"
    # "/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined/fastest"
      )

#raw_path="/media/hcchen/backup/DataCollection/CARLA_0.9.15/carla_combined"
raw_path="/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined"

# 选择排序方法
method="MOT20" # MOT17, MOT20(football), Dance, Carla

# Field model path
#field_model_path="/media/hcchen/data/Trackers/shared_weights/field/field_football.pth"
#field_model_path="/media/hcchen/data/football_extension/carla/training/combined_0/best.pth"
field_model_path="/mnt/12cf47fc-dac6-4abb-a06f-014c3d2e7d30/football_extension/football_combined/training_0/best.pth"

out_path="results_carla/eval_hist_my_${method}.csv"

## 使用預先定義的bounding box(Carla only)
detection_prepared=False

## output_dir
output_dir="football" # football, carla
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
elif [ $method == "Carla" ]; then
    #ckpt="/media/hcchen/data/YOLOX/YOLOX/YOLOX_outputs/yolox_custom/best_ckpt.pth"
    ckpt="pretrained/ocsort_dance_model.pth.tar"
    #exp_file="/media/hcchen/data/YOLOX/YOLOX/exps/example/custom/yolox_custom.py"
    exp_file="exps/example/mot/yolox_dancetrack_test.py"
else
    echo "Invalid method"
    exit 1
fi
sed -i "s|self.exp_file = .*|self.exp_file = \"$exp_file\"|" FieldTrack/config/eval_my.py
sed -i "s|self.ckpt = .*|self.ckpt = \"$ckpt\"|" FieldTrack/config/eval_my.py
sed -i "s|self.field_pretrained = .*|self.field_pretrained = \"$field_model_path\"|" FieldTrack/config/eval_my.py
sed -i "s|self.output_dir = .*|self.output_dir = \"$output_dir\"|" FieldTrack/config/eval_my.py
sed -i "s|self.detection_prepared = .*|self.detection_prepared = $detection_prepared|" FieldTrack/config/eval_my.py



# 循环替换路径并运行 Python 程序
for path in "${paths[@]}"; do
    pred_path="$output_dir/$(basename $path)"

    echo "Running script with path: $path"

    # cd FieldTrack
    # sed -i "s|self.path = .*|self.path = \"$path\"|" config/eval_my.py
    # # 运行修改后的 Python 程序
    # python eval_football_track.py

    # cd ..
    python evaluate.py $raw_path $path "FieldTrack/${pred_path}"
done

# 恢复原始的 main.py 文件
mv FieldTrack/config/eval_my.py.backup FieldTrack/config/eval_my.py