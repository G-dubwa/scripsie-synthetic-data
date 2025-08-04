import os
import numpy as np
import pandas as pd

# --------------------
# CONFIG
# --------------------
NUM_PATIENTS = 100
COUGHS_PER_PATIENT = 3
NUM_FOLDS = 10
IMG_SIZE = (128, 50)  # freq x time
IMAGE_FOLDER = "images"
FOLDS_DIR = "folds"
TTP_FILE = "data/cage/TimeToPositivityDataset.csv"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(FOLDS_DIR, exist_ok=True)
os.makedirs("data/cage", exist_ok=True)

# --------------------
# SPECTROGRAM GENERATION
# --------------------
def generate_synthetic_spectrogram(size=IMG_SIZE):
    """Generate synthetic spectrogram with sinusoidal bands + noise."""
    freq_bins, time_frames = size
    spec = np.random.normal(0, 0.15, size)  # background noise

    # Add harmonics
    num_harmonics = np.random.randint(1, 5)
    for _ in range(num_harmonics):
        freq = np.random.randint(5, freq_bins // 2)
        amp = np.random.uniform(0.5, 2.0)
        time = np.linspace(0, np.pi * np.random.uniform(1, 3), time_frames)
        band = amp * np.sin(time + np.random.uniform(0, 2 * np.pi))
        spec[freq:freq+3, :] += band

    # Normalize to 0–1
    spec = (spec - spec.min()) / (spec.max() - spec.min())
    return spec.astype(np.float32)

# --------------------
# TTP MAPPING (LEARNABLE)
# --------------------
def compute_ttp_from_spectrogram(spec):
    """Map spectrogram mean energy inversely to TTP (1-42)."""
    mean_energy = spec.mean()  # value in [0,1]
    ttp = 42 - (mean_energy * 41)  # higher energy = lower TTP
    ttp += np.random.normal(0, 1)   # slight variation
    return float(np.clip(ttp, 1, 42))

# --------------------
# GENERATE DATASET
# --------------------
patients = [str(i) for i in range(NUM_PATIENTS)]
ttp_values = []

all_entries = []
cough_id_counter = 0

for pid in patients:
    # Generate patient-level TTP (linked to cough energy)
    # We'll average energy of patient's coughs for consistency
    patient_specs = []
    for cough_num in range(COUGHS_PER_PATIENT):
        # Generate spectrogram
        spec = generate_synthetic_spectrogram()
        patient_specs.append(spec)

        # Save to file
        np.save(os.path.join(IMAGE_FOLDER, f"{cough_id_counter}.npy"), spec)

        # Metadata entry
        cough_id = f"{pid}/{cough_num}"
        all_entries.append((cough_id, np.random.choice([0, 1])))
        cough_id_counter += 1

    # Compute TTP based on average energy of patient coughs
    avg_spec = np.mean(patient_specs, axis=0)
    ttp = compute_ttp_from_spectrogram(avg_spec)
    ttp_values.append(ttp)

# Save TTP file
ttp_df = pd.DataFrame({
    "Patient_ID": patients,
    "Time_to_positivity": ttp_values
})
ttp_df.to_csv(TTP_FILE, index=False)

# Split cough metadata into folds
df = pd.DataFrame(all_entries, columns=["Cough_ID", "Status"])
fold_indices = np.array_split(df.index, NUM_FOLDS)
for i, idx in enumerate(fold_indices):
    df.iloc[idx].to_csv(os.path.join(FOLDS_DIR, f"fold_{i}.csv"), index=False)