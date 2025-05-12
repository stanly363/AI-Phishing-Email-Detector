import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --- 1. Load CSV ---
csv_path = "phishing.csv"
df = pd.read_csv(csv_path)

# --- 2. Combine subject & body ---
df["text"] = df["subject"].fillna("") + "\n" + df["body"].fillna("")

# --- 3. Inspect labels ---
print("Unique labels:", df["label"].unique())

# --- 4. Map labels to integers ---
df["mapped_label"] = df["label"].astype(int)

# --- 5. Stratified train/validation split ---
train_df, val_df = train_test_split(
    df[["text", "mapped_label"]],
    test_size=0.2,
    random_state=42,
    stratify=df["mapped_label"]
)

# --- 6. Build tf.data.Dataset pipelines ---
batch_size = 64
def df_to_dataset(dframe, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (dframe["text"].values, dframe["mapped_label"].values)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dframe))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = df_to_dataset(train_df, shuffle=True)
val_ds   = df_to_dataset(val_df,   shuffle=False)

# --- 7. Text vectorization ---
max_vocab_size = 20000
max_seq_length = 500

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_vocab_size,
    output_mode="int",
    output_sequence_length=max_seq_length
)
vectorize_layer.adapt(train_df["text"].values)

def preprocess(text, label):
    return vectorize_layer(text), label

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)

# --- 8. Define & compile the model ---
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab_size, 128),  # removed deprecated input_length
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# --- 9. Check for GPU availability ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPUs available:", [gpu.name for gpu in gpus])
else:
    print("No GPU found, running on CPU")

# --- 10. Set up early stopping & checkpointing ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",   # must end in .keras
        monitor="val_accuracy",
        save_best_only=True
    )
]

# --- 11. Train with callbacks ---
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks
)
