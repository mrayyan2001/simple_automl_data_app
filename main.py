import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
from capstone import *
import matplotlib.pyplot as plt

NUMBER_OF_STEPS = 5


def previous():
    if st.session_state["step_number"] > 1:
        st.session_state["step_number"] -= 1


def next():
    if st.session_state["step_number"] < NUMBER_OF_STEPS:
        st.session_state["step_number"] += 1


if "step_number" not in st.session_state:
    st.session_state["step_number"] = 1

st.title("Step " + str(st.session_state["step_number"]))
# Navigation buttons
col1, col2 = st.columns([9, 1])
if st.session_state["step_number"] not in [1]:  # previous button
    col1.button("Previous", on_click=previous)
if st.session_state["step_number"] not in [1, NUMBER_OF_STEPS]:  # next button
    col2.button("Next", on_click=next)
# Pages
if st.session_state["step_number"] == 1:  # step 1 data uploading
    st.session_state.clear()
    st.session_state["step_number"] = 1

    st.write("# Load Your Dataset")
    path = None
    path = st.file_uploader(
        "dataset_uploading",
        label_visibility="hidden",
        type="csv",
    )
    if path is not None:
        st.session_state["step_number"] += 1
        st.session_state["dataset"] = get_data(path)
        st.rerun()
    pass
elif st.session_state["step_number"] == 2:  # step 2 EDA and handle missing values
    st.write("# Your Dataset")
    if "dataset" in st.session_state:
        st.write(st.session_state["dataset"])
        info = get_info(st.session_state["dataset"])
        st.markdown("---")
        col = st.columns(2)
        col[0].write("## Info Before Cleaning")
        col[0].write(info[0])
        col[0].write(info[1])

        st.session_state["cleaned_dataset"] = clean_data(st.session_state["dataset"])
        info = get_info(st.session_state["cleaned_dataset"])
        col[1].write("## Info After Cleaning")
        col[1].write(info[0])
        col[1].write(info[1])
    else:
        st.write("Go to the previous step and upload your data.")

    x = st.selectbox(
        "x",
        options=st.session_state["dataset"].select_dtypes(np.number).columns,
    )
    y = st.selectbox(
        "y",
        options=st.session_state["dataset"].select_dtypes(np.number).columns,
    )
    st.scatter_chart(data=st.session_state["cleaned_dataset"], x=x, y=y)

elif st.session_state["step_number"] == 3:  # step 3 Encoding
    st.session_state["target"] = []

    st.session_state["target"] = st.multiselect(
        label="select features",
        options=st.session_state["dataset"].select_dtypes(["object", bool]).columns,
    )

    is_ordinal = []
    for i in st.session_state["target"]:
        is_ordinal.append(
            st.radio(
                i,
                options=["Ordinal", "Nominal"],
                horizontal=True,
            )
        )
    is_ordinal = list(map(lambda x: True if x == "Ordinal" else False, is_ordinal))
    st.write("## Data after encoding")
    st.session_state["encoded_dataset"] = st.session_state["cleaned_dataset"].copy()
    st.session_state["encoded_dataset"] = encoding_data(
        st.session_state["encoded_dataset"],
        st.session_state["target"],
        is_ordinal,
    )
    st.write(st.session_state["encoded_dataset"])
elif st.session_state["step_number"] == 4:  # step 4 model training
    st.session_state["x"] = st.multiselect(
        "Select features",
        options=st.session_state["encoded_dataset"]
        .select_dtypes([np.number, bool])
        .columns,
    )
    st.session_state["y"] = st.multiselect(
        "Select label",
        options=st.session_state["encoded_dataset"].columns.drop(st.session_state["x"]),
        max_selections=1,
    )
    st.write(
        st.session_state["encoded_dataset"][
            st.session_state["x"] + st.session_state["y"]
        ]
    )
    if len(st.session_state["y"]) == 1:
        type = st.session_state["dataset"][st.session_state["y"]].dtypes[0]
        st.session_state["is_classification"] = type in [object, bool]
    # if st.session_state["dataset"][y]
elif st.session_state["step_number"] == 5:  # result
    st.write(st.session_state["x"])
    st.write(st.session_state["y"])
    st.write(st.session_state["is_classification"])
    model_train(
        st.session_state["encoded_dataset"],
        st.session_state["is_classification"],
        st.session_state["x"],
        st.session_state["y"],
    )
    pass
