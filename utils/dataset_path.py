path_s4_cvr = [
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Single-source/s4_meta_data.csv",
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Single-source/s4_data/visual_frames",
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Single-source/s4_data/audio_log_mel",
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Single-source/s4_data/gt_masks"
]
path_ms3_cvr = [
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Multi-sources//ms3_meta_data.csv", 
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Multi-sources//ms3_data/visual_frames",
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Multi-sources//ms3_data/audio_log_mel",
    "/data/local_userdata/maoyuxin/datasets/avs/avs_bench_v1/Multi-sources//ms3_data/gt_masks"
]
path_s4_lab = [
    "data/s4_meta_data.csv",
    "data/s4_data/visual_frames",
    "data/s4_data/audio_log_mel",
    "data/s4_data/gt_masks"
]
path_ms3_lab = [
    "/mnt/petrelfs/share_data/maoyuxin/datasets/avs_v1_release/Multi-sources/ms3_meta_data.csv",
    "/mnt/petrelfs/share_data/maoyuxin/datasets/avs_v1_release/Multi-sources/ms3_data/visual_frames",
    "/mnt/petrelfs/share_data/maoyuxin/datasets/avs_v1_release/Multi-sources//ms3_data/audio_log_mel",
    "/mnt/petrelfs/share_data/maoyuxin/datasets/avs_v1_release/Multi-sources//ms3_data/gt_masks"
]


def get_path(flag):
    if flag == "lab":
        path_s4, path_ms3 = path_s4_lab, path_ms3_lab
    elif flag == "cvr":
        path_s4, path_ms3 = path_s4_cvr, path_ms3_cvr
        
    return path_s4, path_ms3
