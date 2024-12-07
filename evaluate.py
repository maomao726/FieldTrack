import motmetrics as mm
import os
import argparse
from football_field.GroundTruthParser import GroundTruthParser
import numpy as np
import pandas as pd
from tqdm import tqdm

def eval(args):
    gt_parser = GroundTruthParser(args.raw_path, args.video_path)
    acc = mm.MOTAccumulator(auto_id=False)
    mh = mm.metrics.create()
    summaries : pd.DataFrame = None

    # DA_acc_list = []
    # DA_on_new_list = []
    # DA_on_appear_list = []

    pred_files = sorted([p for p in os.listdir(args.predict_path) if p.endswith('.txt')])
    for pred_file in tqdm(pred_files):
        #print(f"Processing {pred_file}")
        #
        # id_map = {}
        # appeared_id = []
        # DA_acc_list.append(0)
        # DA_on_new_list.append(0)
        # DA_on_appear_list.append(0)

        pred_total = np.loadtxt(os.path.join(args.predict_path, pred_file), dtype=np.float32, delimiter=',')
        if len(pred_total.shape) == 1:
            pred_total = pred_total.reshape(-1, 4)
        gt_parser.set_tracking_video(pred_file[:-4])
  
        for frameid in range(len(gt_parser.frame_table)):
            pred = pred_total[pred_total[:, 0] == frameid]
            gt_id, pred_id, c = gt_parser.compare_with_gt(frameid, pred[:, [2, 3, 1]])
            acc.update(gt_id, pred_id, c, frameid=frameid)
        
            # events_t = acc.events.reset_index()
            # events = events_t[events_t['FrameId'] == frameid]

            # for _, m in events[events['Type'] == 'MATCH'].iterrows():
            #     if m['HId'] not in id_map:
            #         id_map[m['HId']] = m['OId']
            #     else:
            #         if id_map[m['HId']] != m['OId']:
            #             if m['OId'] not in appeared_id:
            #                 DA_on_new_list[-1] += 1
            #             if m['OId'] not in id_map.values():
            #                 DA_on_appear_list[-1] += 1
            #             DA_acc_list[-1] += 1
            #     if m['OId'] not in appeared_id:
            #         appeared_id.append(m['OId'])
            
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=pred_file[:-4])
        #print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

        if summaries is None:
            summaries = summary
        else:
            summaries = pd.concat([summaries, summary], axis=0)
        acc.reset()


    
    print("------Done------")
    summaries_str = mm.io.render_summary(summaries, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    #print(summaries_str)

    # print("--------DA--------")
    # print(f"DA: {np.sum(DA_acc_list)}, min: {np.min(DA_acc_list)}, max: {np.max(DA_acc_list)}, mean: {np.mean(DA_acc_list)}")
    # print(f"DA_on_new: {np.sum(DA_on_new_list)}, min: {np.min(DA_on_new_list)}, max: {np.max(DA_on_new_list)}, mean: {np.mean(DA_on_new_list)}")
    # print(f"DA_on_appear: {np.sum(DA_on_appear_list)}, min: {np.min(DA_on_appear_list)}, max: {np.max(DA_on_appear_list)}, mean: {np.mean(DA_on_appear_list)}")
    summaries_stats = {
        'video_path' : args.video_path.split('videoData/')[-1],
        'mota' : 1 - (summaries['num_false_positives'].sum() + summaries['num_misses'].sum() + summaries['num_switches'].sum()) / summaries['num_objects'].sum(),
        'idf1' : 2 * summaries['idtp'].sum() / (2 * summaries['idtp'].sum() + summaries['idfp'].sum() + summaries['idfn'].sum()),
        'fp' : summaries['num_false_positives'].sum(),
        'fn' : summaries['num_misses'].sum(),
        'idsw' : summaries['num_switches'].sum(),
        'gt' : summaries['num_objects'].sum(),
        'idp' : summaries['idtp'].sum() / (summaries['idtp'].sum() + summaries['idfp'].sum()),
        'idr' : summaries['idtp'].sum() / (summaries['idtp'].sum() + summaries['idfn'].sum()),
        'idt' : summaries['num_transfer'].sum(),
        'ida' : summaries['num_ascend'].sum(),
        'idm' : summaries['num_migrate'].sum()
    }

    for k, v in summaries_stats.items():
        if isinstance(v, np.float64):
            summaries_stats[k] = round(v, 4)
        #summaries_stats[k] = round(v, 4)
    print(f"Summary Stats for all videos: ")
    print("\n".join([f"{k}: \t {v}" for k, v in summaries_stats.items()]))

   
    # save to csv
    if args.output is not None:
        if os.path.exists(args.output):
            df:pd.DataFrame = pd.read_csv(args.output)
            df = pd.concat([df, pd.DataFrame([summaries_stats], columns=summaries_stats.keys())], ignore_index=True)
        else:
            df = pd.DataFrame([summaries_stats], columns=summaries_stats.keys())
        df.to_csv(args.output, index=False)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=str, help="raw data path, not cutted")
    parser.add_argument("video_path", type=str, help="data path, (maybe)cutted")
    parser.add_argument("predict_path", type=str, help="predict result, directory including many 'xxxxxx.txt'")
    parser.add_argument("--output", type=str, default=None, help="output path, suggest to be a csv file, default is None")

    args = parser.parse_args()
    eval(args)

    