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

    pred_files = [p for p in os.listdir(args.predict_path) if p.endswith('.txt')]
    for pred_file in tqdm(pred_files):
        print(pred_file)
        pred_total = np.loadtxt(os.path.join(args.predict_path, pred_file), dtype=np.float32, delimiter=',')
        
        gt_parser.set_tracking_video(pred_file[:-4])
  
        
        for frameid in range(int(np.max(pred_total[:,0]) + 1)):
            pred = pred_total[pred_total[:, 0] == frameid]
            
            gt_id, pred_id, c = gt_parser.compare_with_gt(frameid, pred[:, [2, 3, 1]])
            acc.update(gt_id, pred_id, c, frameid=frameid)
        
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=pred_file[:-4])
        #print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

        if summaries is None:
            summaries = summary
        else:
            summaries = pd.concat([summaries, summary], axis=0)
        acc.reset()


    
    print("------Done------")
    summaries_str = mm.io.render_summary(summaries, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(summaries_str)

    summaries_stats = {
        'video_path' : args.video_path.split('videoData/')[-1],
        'mota' : 1 - (summaries['num_false_positives'].sum() + summaries['num_misses'].sum() + summaries['num_switches'].sum()) / summaries['num_objects'].sum(),
        'idf1' : 2 * summaries['idtp'].sum() / (2 * summaries['idtp'].sum() + summaries['idfp'].sum() + summaries['idfn'].sum()),
        'fp' : summaries['num_false_positives'].sum(),
        'fn' : summaries['num_misses'].sum(),
        'idsw' : summaries['num_switches'].sum(),
        'gt' : summaries['num_objects'].sum(),
        'idp' : summaries['idtp'].sum() / (summaries['idtp'].sum() + summaries['idfp'].sum()),
        'idr' : summaries['idtp'].sum() / (summaries['idtp'].sum() + summaries['idfn'].sum())
    }

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

    