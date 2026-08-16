import os
import time
configs_list = [
    './configs/cfg_voc21.py',
    './configs/cfg_voc20.py',
    './configs/cfg_context60.py',
    './configs/cfg_context59.py',
    './configs/cfg_coco_object.py',
    './configs/cfg_coco_stuff164k.py',
    './configs/cfg_ade20k.py',
    './configs/cfg_city_scapes.py',
    './configs/cfg_ade847.py',
    './configs/cfg_context459.py',
]

n_gpu = os.environ.get('EVAL_NGPU', '2')   # default 2 GPUs; set EVAL_NGPU for sweeps
for config in configs_list:
    print(f"Running {config}  (GPUS={n_gpu}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','all')})", flush=True)
    # If a dataset fails (e.g. OOM from a foreign process on the same GPU), wait 5 minutes and retry, up to 3 times, so the run leaves no holes
    for attempt in range(1, 4):
        code = os.system(f"bash ./dist_test.sh {config} {n_gpu}")
        if code == 0:
            break
        print(f"[retry {attempt}/3] {config} exited with code {code}, retrying in 5 minutes", flush=True)
        time.sleep(300)
