import os
import json
import cv2
import argparse

def read_frame_by_num(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        #print("Error reading frame")
        raise ValueError("Error reading frame")
    return frame

def main(): # 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='path_to_your/ego4d/hand_object_interactions/v2/')
    args = parser.parse_args()
    dir_clips = os.path.join(args.root_path, "clips")
    dir_annotations = os.path.join(args.root_path, "annotations")
    fho_main = json.load(open(os.path.join(dir_annotations, "fho_main.json"), "rb"))
    dir_outputs = os.path.join(args.root_path, "clips_jpgs/processed")
    info_clips = {}
    for i_video in range(len(fho_main['videos'])):
        annotated_intervals = fho_main['videos'][i_video]['annotated_intervals']
        print("len(annotated_intervals):", len(annotated_intervals))
        for i_clip in range(len(annotated_intervals)):
            print("processing i_video:{}, i_clip:{}".format(i_video, i_clip))
            clip_uid = annotated_intervals[i_clip]['clip_uid']
            narrated_actions = annotated_intervals[i_clip]['narrated_actions']
            if not os.path.exists(os.path.join(dir_clips, f'{clip_uid}.mp4')):
                continue
            cap = cv2.VideoCapture(os.path.join(dir_clips, f'{clip_uid}.mp4'))
            info_multi_action = []
            if len(narrated_actions)==0:
                continue
            for i_action in range(len(narrated_actions)):
                info_action = {}
                narrated_action = narrated_actions[i_action]
                narration_text = narrated_action['narration_text'].strip()
                if narrated_action['clip_critical_frames'] is None:
                    continue
                pre_frame_num = narrated_action['clip_critical_frames']['pre_frame']
                pnr_frame_num = narrated_action['clip_critical_frames']['pnr_frame']
                post_frame_num = narrated_action['clip_critical_frames']['post_frame']
                pre_frame = read_frame_by_num(cap, pre_frame_num)
                pnr_frame = read_frame_by_num(cap, pnr_frame_num)
                post_frame = read_frame_by_num(cap, post_frame_num)
                if narrated_action['frames'] is not None:
                    frame_types = [narrated_action['frames'][i_frame]['frame_type'] for i_frame in range(len(narrated_action['frames']))]
                    if 'pre_frame' in frame_types:
                        pre_boxes = narrated_action['frames'][frame_types.index('pre_frame')]['boxes']
                    else:
                        pre_boxes = []
                    if 'pnr_frame' in frame_types:
                        pnr_boxes = narrated_action['frames'][frame_types.index('pnr_frame')]['boxes']
                    else:
                        pnr_boxes = []
                    if 'post_frame' in frame_types:
                        post_boxes = narrated_action['frames'][frame_types.index('post_frame')]['boxes']
                    else:
                        post_boxes = []
                else:
                    pre_boxes = []
                    pnr_boxes = []
                    post_boxes = []
                save_dir = os.path.join(clip_uid, 'action_{}_'.format(str(i_action).zfill(3)) + '_'.join(narration_text.split(' ')))
                dir_pre_frame = os.path.join(save_dir, 'pre_frame.jpg')
                dir_pnr_frame = os.path.join(save_dir, 'pnr_frame.jpg')
                dir_post_frame = os.path.join(save_dir, 'post_frame.jpg')
                os.makedirs(os.path.join(dir_outputs, save_dir), exist_ok = True)
                cv2.imwrite(os.path.join(dir_outputs, dir_pre_frame), pre_frame)
                cv2.imwrite(os.path.join(dir_outputs, dir_pnr_frame), pnr_frame)
                cv2.imwrite(os.path.join(dir_outputs, dir_post_frame), post_frame)
                info_action = {
                    'narration_text': narration_text,
                    'pre_frame': {'frame_num': pre_frame_num, 'boxes':pre_boxes, 'path': dir_pre_frame},
                    'pnr_frame': {'frame_num': pnr_frame_num, 'boxes':pnr_boxes, 'path': dir_pnr_frame},
                    'post_frame': {'frame_num': post_frame_num, 'boxes':post_boxes, 'path': dir_post_frame},
                }
                info_multi_action.append(info_action)
                
            info_clips[clip_uid] = info_multi_action
            cap.release()
    json.dump(info_clips, open(os.path.join(dir_outputs, 'info_clips.json'), 'w'))

if __name__ == "__main__":
    main()
        