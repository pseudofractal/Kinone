import json
import time

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

st.title("Training Dashboard")

log_file = "training_log.json"

st.header("Metrics")
col1, col2, col3, col4 = st.columns(4)
epoch_metric = col1.empty()
loss_metric = col2.empty()
auc_metric = col3.empty()
accuracy_metric = col4.empty()

st.header("Charts")
loss_chart = st.empty()
auc_chart = st.empty()
accuracy_chart = st.empty()
lr_chart = st.empty()

while True:
    try:
        with open(log_file, "r") as f:
            log_data = [json.loads(line) for line in f]

        if log_data:
            df = pd.DataFrame(log_data)

            # Update metrics
            latest_epoch = df['epoch'].iloc[-1]
            latest_loss = df['loss'].iloc[-1]
            latest_auc = df['val_auc'].iloc[-1]
            latest_accuracy = df['val_accuracy'].iloc[-1]

            epoch_metric.metric("Epoch", f"{latest_epoch}/{df['epoch'].max()}")
            loss_metric.metric("Training Loss", f"{latest_loss:.4f}")
            auc_metric.metric("Validation AUC", f"{latest_auc:.4f}")
            accuracy_metric.metric("Validation Accuracy", f"{latest_accuracy:.4f}")

            # Update charts
            loss_chart.line_chart(df, x='epoch', y='loss', use_container_width=True)
            auc_chart.line_chart(df, x='epoch', y='val_auc', use_container_width=True)
            accuracy_chart.line_chart(df, x='epoch', y='val_accuracy', use_container_width=True)
            lr_chart.line_chart(df, x='epoch', y='lr', use_container_width=True)

    except FileNotFoundError:
        st.info(f"Waiting for training to start... Check if {log_file} exists. It will only do so after 1 epoch.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    time.sleep(5)
