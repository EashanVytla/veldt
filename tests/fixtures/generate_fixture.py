#!/usr/bin/env python3
"""Generate mini_lerobot v3 test fixture parquet files.

2 episodes of 45 frames each (90 total), 30fps, concatenated in one video file.
Episode 0: frames 0-44 (timestamps 0.0 - 1.467s in video)
Episode 1: frames 45-89 (timestamps 1.5 - 2.967s in video)
"""
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "mini_lerobot")
FPS = 30
EPISODES = 2
FRAMES_PER_EP = 45
TOTAL_FRAMES = EPISODES * FRAMES_PER_EP

np.random.seed(42)

# === data/chunk-000/file-000.parquet ===
# All 90 frames in one file
episode_indices = []
frame_indices = []
timestamps = []
global_indices = []
task_indices = []
next_done = []
states = []
actions = []

for ep in range(EPISODES):
    for f in range(FRAMES_PER_EP):
        global_idx = ep * FRAMES_PER_EP + f
        episode_indices.append(ep)
        frame_indices.append(f)
        timestamps.append(f / FPS)
        global_indices.append(global_idx)
        task_indices.append(0)
        next_done.append(f == FRAMES_PER_EP - 1)
        states.append(np.random.randn(4).astype(np.float32).tolist())
        actions.append(np.random.randn(4).astype(np.float32).tolist())

data_table = pa.table({
    "observation.state": pa.array(states, type=pa.list_(pa.float32())),
    "action": pa.array(actions, type=pa.list_(pa.float32())),
    "episode_index": pa.array(episode_indices, type=pa.int64()),
    "frame_index": pa.array(frame_indices, type=pa.int64()),
    "timestamp": pa.array(timestamps, type=pa.float32()),
    "next.done": pa.array(next_done, type=pa.bool_()),
    "index": pa.array(global_indices, type=pa.int64()),
    "task_index": pa.array(task_indices, type=pa.int64()),
})

data_path = os.path.join(FIXTURE_DIR, "data", "chunk-000", "file-000.parquet")
pq.write_table(data_table, data_path)
print(f"Wrote {data_path} ({len(data_table)} rows)")

# === meta/episodes/chunk-000/file-000.parquet ===
# 2 episodes, both in chunk-000/file-000 for data and video
episodes_table = pa.table({
    "episode_index": pa.array([0, 1], type=pa.int64()),
    "data/chunk_index": pa.array([0, 0], type=pa.int64()),
    "data/file_index": pa.array([0, 0], type=pa.int64()),
    "dataset_from_index": pa.array([0, 45], type=pa.int64()),
    "dataset_to_index": pa.array([45, 90], type=pa.int64()),
    "videos/observation.images.top/chunk_index": pa.array([0, 0], type=pa.int64()),
    "videos/observation.images.top/file_index": pa.array([0, 0], type=pa.int64()),
    "videos/observation.images.top/from_timestamp": pa.array([0.0, 1.5], type=pa.float64()),
    "videos/observation.images.top/to_timestamp": pa.array([1.5, 3.0], type=pa.float64()),
    "length": pa.array([45, 45], type=pa.int64()),
    "meta/episodes/chunk_index": pa.array([0, 0], type=pa.int64()),
    "meta/episodes/file_index": pa.array([0, 0], type=pa.int64()),
    "tasks": pa.array([["test_task"], ["test_task"]], type=pa.list_(pa.string())),
})

episodes_path = os.path.join(FIXTURE_DIR, "meta", "episodes", "chunk-000", "file-000.parquet")
pq.write_table(episodes_table, episodes_path)
print(f"Wrote {episodes_path} ({len(episodes_table)} rows)")

# === meta/tasks.parquet ===
tasks_table = pa.table({
    "task_index": pa.array([0], type=pa.int64()),
    "task": pa.array(["test_task"], type=pa.string()),
})

tasks_path = os.path.join(FIXTURE_DIR, "meta", "tasks.parquet")
pq.write_table(tasks_table, tasks_path)
print(f"Wrote {tasks_path} ({len(tasks_table)} rows)")

print("Done!")
