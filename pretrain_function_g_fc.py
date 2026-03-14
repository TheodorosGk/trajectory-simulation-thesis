import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from generator.function_g_fc import FunctionGFC
from utils.ListDataset import ListDataset
from utils.utils import get_logger

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# This script pretrains the road-level FunctionGFC generator.
# Replace the paths below with your own local paths before running.

# =========================================================
# USER CONFIGURATION
# =========================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_model = True

max_epoch = 10
batch_size = 64
learning_rate = 0.0005
weight_decay = 0.00001
lr_patience = 2
lr_decay_ratio = 0.1
early_stop_lr = 1e-6
clip = 5.0

# Path to the Porto pretrain_data folder
data_root = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

# Path to the Porto save folder
save_root = r"PASTE_YOUR_MODEL_SAVE_PATH_HERE"

save_folder = save_root
save_file_name = "function_g_fc.pt"
temp_folder = os.path.join(save_root, "temp_function_g")

# =========================================================
# DATASET CONFIGURATION
# =========================================================
road_num = 11095
time_size = 2880
loc_pad = road_num
time_pad = time_size

data_feature = {
    "road_num": road_num + 1,
    "time_size": time_size + 1,
    "road_pad": loc_pad,
    "time_pad": time_pad
}

gen_config = {
    "road_emb_size": 256,
    "time_emb_size": 50,
    "hidden_size": 256,
    "dropout_p": 0.6,
    "lstm_layer_num": 2,
    "pretrain_road_rep": None,
    "dis_weight": 0.5,
    "device": device
}

logger = get_logger(name="FunctionGFC")
logger.info("Reading data")

# =========================================================
# LOAD TRAIN / EVAL / TEST DATA
# =========================================================
train_csv = os.path.join(data_root, "porto_taxi_pretrain_input_train.csv")
eval_csv = os.path.join(data_root, "porto_taxi_pretrain_input_eval.csv")
test_csv = os.path.join(data_root, "porto_taxi_pretrain_input_test.csv")

train_data = pd.read_csv(train_csv).values.tolist()
eval_data = pd.read_csv(eval_csv).values.tolist()
test_data = pd.read_csv(test_csv).values.tolist()

train_num = len(train_data)
eval_num = len(eval_data)
test_num = len(test_data)
total_data = train_num + eval_num + test_num

logger.info(
    "Total input records: {}. Train: {}, Val: {}, Test: {}".format(
        total_data, train_num, eval_num, test_num
    )
)

train_dataset = ListDataset(train_data)
eval_dataset = ListDataset(eval_data)
test_dataset = ListDataset(test_data)


def collate_fn(indices):
    """
    Custom DataLoader collation function for FunctionGFC pretraining.
    """
    batch_trace_loc = []
    batch_trace_time = []
    batch_des = []
    batch_candidate_set = []
    batch_candidate_dis = []
    batch_target = []

    trace_loc_len = []
    candidate_set_len = []

    for item in indices:
        trace_loc = [int(i) for i in item[0].split(",")]
        trace_time = [int(i) for i in item[1].split(",")]
        destination = item[2]
        candidate_set = [int(i) for i in item[3].split(",")]
        candidate_dis = [float(i) for i in item[4].split(",")]
        target = item[5]

        batch_trace_loc.append(trace_loc)
        batch_trace_time.append(trace_time)
        batch_des.append(destination)
        batch_candidate_set.append(candidate_set)
        batch_candidate_dis.append(candidate_dis)
        batch_target.append(target)

        trace_loc_len.append(len(trace_loc))
        candidate_set_len.append(len(candidate_set))

    max_trace_len = max(trace_loc_len)
    max_candidate_size = max(candidate_set_len)

    for i in range(len(batch_trace_loc)):
        pad_len = max_trace_len - len(batch_trace_loc[i])
        batch_trace_loc[i] += [loc_pad] * pad_len
        batch_trace_time[i] += [time_pad] * pad_len

        while len(batch_candidate_set[i]) < max_candidate_size:
            assert len(batch_candidate_set[i]) != 1, "Candidate set size is 1"
            pad_index = np.random.randint(len(batch_candidate_set[i]))
            if pad_index != batch_target[i]:
                batch_candidate_set[i].append(batch_candidate_set[i][pad_index])
                batch_candidate_dis[i].append(batch_candidate_dis[i][pad_index])

    return [
        torch.LongTensor(batch_trace_loc).to(device),
        torch.LongTensor(batch_trace_time).to(device),
        torch.LongTensor(batch_des).to(device),
        torch.LongTensor(batch_candidate_set).to(device),
        torch.FloatTensor(batch_candidate_dis).to(device),
        torch.LongTensor(batch_target).to(device),
    ]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# =========================================================
# INIT MODEL
# =========================================================
gen_model = FunctionGFC(gen_config, data_feature).to(device)

logger.info("Initialized FunctionGFC")
logger.info(gen_model)

optimizer = torch.optim.Adam(
    gen_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",
    patience=lr_patience,
    factor=lr_decay_ratio
)

# =========================================================
# TRAIN / LOAD MODEL
# =========================================================
if train_model:
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    metrics = []

    for epoch in range(max_epoch):
        logger.info(f"Starting train epoch {epoch}")

        gen_model.train(True)
        train_loss = 0

        for trace_loc, trace_time, des, candidate_set, candidate_dis, target in tqdm(train_loader, desc="Training model"):
            optimizer.zero_grad()

            trace_mask = ~(trace_loc == loc_pad)

            loss = gen_model.calculate_loss(
                trace_loc=trace_loc,
                trace_time=trace_time,
                des=des,
                candidate_set=candidate_set,
                candidate_dis=candidate_dis,
                target=target,
                trace_mask=trace_mask
            )

            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), clip)
            optimizer.step()

        val_hit = 0
        gen_model.train(False)

        for trace_loc, trace_time, des, candidate_set, candidate_dis, target in tqdm(val_loader, desc="Validating model"):
            trace_mask = ~(trace_loc == loc_pad)

            score = gen_model.predict_g(
                trace_loc=trace_loc,
                trace_time=trace_time,
                des=des,
                candidate_set=candidate_set,
                candidate_dis=candidate_dis,
                trace_mask=trace_mask
            )

            target = target.tolist()
            _, index = torch.topk(score, 1, dim=1)

            for i, pred in enumerate(index):
                if target[i] in pred:
                    val_hit += 1

        val_ac = val_hit / eval_num
        metrics.append(val_ac)
        lr_scheduler.step(val_ac)

        torch.save(gen_model.state_dict(), os.path.join(temp_folder, f"function_g_{epoch}.pt"))

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "==> Train Epoch {}: Train Loss {:.6f}, Val Acc {:.6f}, lr {}".format(
                epoch, train_loss, val_ac, lr
            )
        )

        if lr < early_stop_lr:
            logger.info("Early stop triggered")
            break

    best_epoch = np.argmax(metrics)
    load_temp_file = f"function_g_{best_epoch}.pt"
    logger.info(f"Loading best model from epoch {best_epoch}")
    gen_model.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file), map_location=device))

else:
    gen_model.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))

# =========================================================
# TEST MODEL
# =========================================================
test_hit = 0
gen_model.train(False)

for trace_loc, trace_time, des, candidate_set, candidate_dis, target in tqdm(test_loader, desc="Testing model"):
    trace_mask = ~(trace_loc == loc_pad)

    score = gen_model.predict_g(
        trace_loc=trace_loc,
        trace_time=trace_time,
        des=des,
        candidate_set=candidate_set,
        candidate_dis=candidate_dis,
        trace_mask=trace_mask
    )

    target = target.tolist()
    _, index = torch.topk(score, 1, dim=1)

    for i, pred in enumerate(index):
        if target[i] in pred:
            test_hit += 1

test_ac = test_hit / test_num
logger.info("==> Test Result: acc {:.6f}".format(test_ac))

# =========================================================
# SAVE FINAL MODEL
# =========================================================
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

torch.save(gen_model.state_dict(), os.path.join(save_folder, save_file_name))

# =========================================================
# CLEAN TEMP FILES
# =========================================================
if os.path.exists(temp_folder):
    for root, dirs, files in os.walk(temp_folder):
        for name in files:
            remove_path = os.path.join(root, name)
            os.remove(remove_path)