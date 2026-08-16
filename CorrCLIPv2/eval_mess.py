import os
import time
configs_list = [
    './configs/mess/cfg_bdd100k.py',
    './configs/mess/cfg_dark_zurich.py',
    './configs/mess/cfg_mhp.py',
    './configs/mess/cfg_foodseg103.py',
    './configs/mess/cfg_atlantis.py',
    './configs/mess/cfg_dram.py',
    './configs/mess/cfg_isaid.py',
    './configs/mess/cfg_potsdam.py',
    './configs/mess/cfg_worldfloods.py',
    './configs/mess/cfg_floodnet.py',
    './configs/mess/cfg_uavid.py',
    './configs/mess/cfg_kvasir.py',
    './configs/mess/cfg_chase_db1.py',
    './configs/mess/cfg_cryonuseg.py',
    './configs/mess/cfg_paxray_bones.py',
    './configs/mess/cfg_paxray_diaphragm.py',
    './configs/mess/cfg_paxray_lung.py',
    './configs/mess/cfg_paxray_mediastinum.py',
    './configs/mess/cfg_corrosion.py',
    './configs/mess/cfg_deepcrack.py',
    './configs/mess/cfg_zerowaste.py',
    './configs/mess/cfg_pst900.py',
    './configs/mess/cfg_suim.py',
    './configs/mess/cfg_cub.py',
    './configs/mess/cfg_cwfid.py',
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
