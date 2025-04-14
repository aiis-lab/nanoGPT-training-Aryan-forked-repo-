import json
import numpy as np 

#load the file
baseline_file = r'C:\Users\aiis-\Desktop\q-norm\nanoGPT-training-Aryan-forked-repo-\out-shakespeare-char\logs\baseline_run_1743974308.json'
with open(baseline_file, 'r') as f:
    data = json.load(f)

# Extract data
iters = data['iterations']
train_losses = data['train_loss']
val_losses = [v['value'] for v in data['val_loss']]
val_iters = [v['iteration'] for v in data['val_loss']]
times = data['times']

# 1. Iteration Efficiency
val_diffs = np.diff(val_losses)
conv_idx = next((i for i, diff in enumerate(val_diffs) if abs(diff) < 0.01), len(val_losses) - 1)
conv_iter = val_iters[conv_idx]
print(f"Iteration Efficiency: Converged at iteration {conv_iter}")

# 2. Training Stability
train_std = np.std(train_losses)
print(f"Training Stability: Std Dev of Train Loss = {train_std:.4f}")

# 3. Final Losses
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]
print(f"Final Train Loss: {final_train_loss:.4f}")
print(f"Final Val Loss: {final_val_loss:.4f}")

# 4. Runtime Performance
mean_time = np.mean(times[1:])  # Exclude first iter (3.66s outlier)
total_time = np.sum(times) / 60  # Total runtime in minutes
print(f"Mean Iteration Time: {mean_time:.4f} seconds")
print(f"Total Runtime: {total_time:.2f} minutes")

sample_file = r'C:\Users\aiis-\Desktop\q-norm\nanoGPT-training-Aryan-forked-repo-\out-shakespeare-char\logs\samples_1743975646.json'
vocab = " \n!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
try:
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)
    token_ids = sample_data['samples'][0]  # First sequence
    completion = ''.join([vocab[t] for t in token_ids if t < len(vocab)])
    print(f"Sample Completion (Temp 0.7): {completion}")
except FileNotFoundError:
    print("Sample file not found. Using provided sample data.")
    # Use your provided sample directly
    token_ids = [0, 0, 15, 50, 53, 61, 52, 10, 0, 31, 53, 1, 63, 53, 59, 1, 61, 47, 50, 50, 1, 40, 43, 1, 57, 53, 6, 1, 57, 47, 56, 6, 1, 39, 1, 61, 53, 56, 42, 1, 53, 44, 1, 40, 50, 53, 53, 42, 6, 1, 39]
    completion = ''.join([vocab[t] for t in token_ids if t < len(vocab)])
    print(f"Sample Completion (Temp 0.7): {completion}")