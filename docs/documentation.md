# Documentation

## Overview

This repository contains the code and helper scripts used in the thesis:

**Simulation of Multiple-Aspect Trajectories with Machine Learning Methods**

The implementation is based on the **TS-TrajGen** pipeline and focuses on the **Porto Taxi** dataset.

The main workflow is:

1. Prepare the road-level input files
2. Build pretraining datasets
3. Pretrain the road-level models
4. Build the region-level data and features
5. Pretrain the region-level models
6. Optionally run adversarial training
7. Generate synthetic trajectories for the test set
8. Evaluate the generated trajectories

---

## 1. Required Input Files

To run the Porto Taxi pipeline, the main required files are:

### Map-matched trajectory files
- `porto_mm_train.csv`
- `porto_mm_test.csv`

### Road network files
- `porto.geo`
- `porto.rel`

These files are the starting point for the preprocessing and model pipeline.

---

## 2. Main File Formats

### 2.1 `porto_mm_train.csv` / `porto_mm_test.csv`

Each row corresponds to one map-matched trajectory.

Important columns:

- `mm_id`: map-matched record id
- `user_id`: vehicle id
- `traj_id`: trajectory id
- `rid_list`: sequence of road segment ids
- `time_list`: sequence of timestamps for the corresponding road segments

### 2.2 `porto.geo`

Contains the road segments of the network.

Typical information includes:
- road segment id
- geometry
- length
- road type
- start / end node
- one-way information

### 2.3 `porto.rel`

Contains the directed connectivity of the road network.

It defines which road segment can be followed by which next segment.

---

## 3. Step-by-Step Execution

## Step 1 — Build pretraining inputs

Run the preprocessing script that converts map-matched trajectories into supervised next-step prediction samples.

### Script
- `scripts_for_files/preprocess_pretrain_input_Porto.py`

### Input
- `porto_mm_train.csv`
- `porto_mm_test.csv`
- `porto.geo`
- `porto.rel`

### What it does
- builds road adjacency information
- builds representative GPS coordinates for road segments
- encodes timestamps
- converts trajectories into `(prefix -> next hop)` training samples

### Main outputs
- `adjacent_list.json`
- `porto_rid_gps.json`
- `porto_taxi_pretrain_input_train.csv`
- `porto_taxi_pretrain_input_eval.csv`
- `porto_taxi_pretrain_input_test.csv`

---

## Step 2 — Pretrain road-level Function H

Pretrain the graph-based scorer used to evaluate candidate next road segments.

### Script
- `pretrain_gat_fc.py`

### Input
- `porto_taxi_pretrain_input_train.csv`
- `porto_taxi_pretrain_input_eval.csv`
- `porto_taxi_pretrain_input_test.csv`
- `porto_adjacent_mx.npz`
- `porto_node_feature.pt`

### Output
- `gat_fc.pt`

---

## Step 3 — Pretrain road-level Function G

Pretrain the sequence-based next-hop policy model.

### Script
- `pretrain_function_g_fc.py`

### Input
- `porto_taxi_pretrain_input_train.csv`
- `porto_taxi_pretrain_input_eval.csv`
- `porto_taxi_pretrain_input_test.csv`

### Output
- `function_g_fc.pt`

---

## Step 4 — Prepare region partition input

Convert the road graph to KaHIP-compatible format.

### Script
- `process_kahip_graph_format.py`

### Input
- `porto.geo`
- `porto.rel`

### Output
- graph file for KaHIP / KaFFPa partitioning

---

## Step 5 — Process region partition results

After running graph partitioning externally, convert the result into region mappings.

### Script
- `process_kaffpa_res.py`

### Input
- KaHIP / KaFFPa partition output
- reindexed road graph information

### Output
- `rid2region.json`
- `region2rid.json`

---

## Step 6 — Build region adjacency

Construct the region-level graph.

### Script
- `construct_traffic_zone_relation.py`

### Input
- `porto.rel`
- `rid2region.json`
- `region2rid.json`

### Output
- `region_adj_mx.npz`
- `region_adjacent_list.json`

---

## Step 7 — Map road trajectories to region trajectories

Convert road-level trajectories into region-level trajectories.

### Script
- `map_region_traj.py`

### Input
- map-matched road trajectories
- `rid2region.json`

### Output
- region-level train/eval/test trajectory files

---

## Step 8 — Encode region pretraining inputs

Create supervised next-step samples at region level.

### Script
- `encode_region_traj.py`

### Input
- region trajectory files
- `region_adjacent_list.json`
- region center / GPS data

### Output
- `region_pretrain_input_train.csv`
- `region_pretrain_input_eval.csv`
- `region_pretrain_input_test.csv`

---

## Step 9 — Prepare region features

Build region node features using road-level learned representations.

### Script
- `prepare_region_feature.py`

### Input
- road-level pretrained model
- `rid2region.json`
- `region2rid.json`

### Output
- `region_feature.pt`

---

## Step 10 — Pretrain region-level models

Pretrain the region-level Function G and Function H models.

### Main scripts
- `pretrain_region_function_g_fc.py`
- `pretrain_region_gat_fc.py`

### Output
- region-level pretrained checkpoints

---

## Step 11 — Optional adversarial training

Run adversarial learning for road level and/or region level.

### Scripts
- `train_gan.py`
- `train_region_gan.py`

### Output
- updated generator and discriminator checkpoints
- training logs
- similarity metric logs

This stage is optional if the goal is only to reproduce the main pretrained generation pipeline.

---

## Step 12 — Final trajectory generation

Generate synthetic trajectories for the test set.

### Script
- `our_model_generate.py`

### Input
- `porto_mm_test.csv`
- pretrained road-level checkpoints
- pretrained region-level checkpoints

### What it uses
For each test trajectory:
- origin
- destination
- target length

### Output
Generated trajectories in CSV format, typically with:
- `traj_id`
- `rid_list`
- `time_list`

---

## 4. Important Intermediate Files

### Road level
- `adjacent_list.json`: valid road-to-road transitions
- `porto_rid_gps.json`: representative coordinates for each road segment
- `porto_adjacent_mx.npz`: road adjacency matrix
- `porto_node_feature.pt`: road node features

### Region level
- `rid2region.json`: road id to region id
- `region2rid.json`: region id to list of roads
- `region_adj_mx.npz`: region adjacency matrix
- `region_adjacent_list.json`: valid region-to-region transitions
- `region_feature.pt`: region node features

---

## 5. Main Checkpoints Produced

### Road-level checkpoints
- `gat_fc.pt`
- `function_g_fc.pt`

### Region-level checkpoints
- region-level Function G checkpoint
- region-level Function H checkpoint

### Optional GAN checkpoints
- generator checkpoint
- discriminator checkpoint

---

## 6. Minimal Execution Order

If someone wants only the core pipeline, the simplest execution order is:

1. `preprocess_pretrain_input_Porto.py`
2. `pretrain_gat_fc.py`
3. `pretrain_function_g_fc.py`
4. `process_kahip_graph_format.py`
5. run KaHIP / KaFFPa externally
6. `process_kaffpa_res.py`
7. `construct_traffic_zone_relation.py`
8. `map_region_traj.py`
9. `encode_region_traj.py`
10. `prepare_region_feature.py`
11. `pretrain_region_gat_fc.py`
12. `pretrain_region_function_g_fc.py`
13. `our_model_generate.py`

---

## 7. Notes

- Some scripts may contain hard-coded paths and may need path updates before execution.
- Large datasets and large generated files are not necessarily included in the repository.
- Exact reproduction depends on having the correct intermediate files and folder structure.
- KaHIP / KaFFPa partitioning is an external step and must be run separately before processing the region partition results.

---

## 8. Final Remark

This documentation is intended to help a reader understand the Porto Taxi pipeline and reproduce the main execution steps of the thesis implementation with minimal confusion.
