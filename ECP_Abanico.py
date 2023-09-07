import pandas as pd
import matplotlib.pyplot as plt
import lasio
import numpy as np
import seaborn as sns
import streamlit as st
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(
    page_title="ECP-Abanico K-Phi Project",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ECOPETROL")
st.subheader("Abanico Project")

appMode = st.sidebar.selectbox(
    "App Mode",
    [
        "Single Well Logs",
        "Multi Well Logs",
        "Core Data",
        "Pore Throat Radius",
        "Reference Data",
    ],
)

headerNames = [
    "Depth",
    "Porosity 800",
    "K_air 800",
    "R45 Calculado",
    "Rock Type",
    "PHI*K_air",
]

if appMode == "Single Well Logs":
    wellName = st.sidebar.selectbox(
        "Well Name", ("Abanico-2", "Abanico-19", "Abanico-34")
    )

    wellSummary, statsTab, plotTab, crossPlot, dataTab = st.tabs(
        ["Well Summary", "Stats", "Plot", "CrossPlots", "Data"]
    )

    if wellName == "Abanico-2":
        fileName = "ABANICO-2_LAS_K.las"
    elif wellName == "Abanico-19":
        fileName = "ABANICO-19_LAS_K.las"
    elif wellName == "Abanico-34":
        fileName = "ABANICO-34_LAS_K.las"

    # Load .las files and create a dataframe
    df = lasio.read("../data/wells/" + fileName).df()
    # Move the index to a column, label Depth
    df = df.reset_index()

    # Take the column names and put them in a list, except the first
    wellLogList = df.columns[1:].tolist()

    # Display data in dataTab
    with wellSummary:
        st.write("Well Name:", wellName)
        # Take the min and max values of the Depth column
        st.write("Min Depth:", df["DEPTH"].min())
        st.write("Max Depth:", df["DEPTH"].max())

    with statsTab:
        st.table(df.describe())

    # Display data in dataTab
    with dataTab:
        st.table(df)

    with plotTab:
        # In the sidebar, show a selectbox options

        xAxis = st.selectbox("Well Log", wellLogList, index=1)
        showLitho = st.radio("Show Lithology", ["Yes", "No"])
        # Plot the DEPTH column in the x axis and the yAxis selected in the y axis using plotly
        fig = px.line(df, x=xAxis, y="DEPTH")
        # Invert y axis
        fig.update_yaxes(autorange="reversed")

        if showLitho == "Yes":
            if wellName == "Abanico-2":
                # Draw a rectagle in the plot, semi transparent, from y = 2561.6 till y = 2627.7, that coverets the entire horizontal size of the plot, show the legend and name of each
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2561.6,
                    x1=1,
                    y1=2627.7,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2627.7,
                    x1=1,
                    y1=2734.7,
                    fillcolor="Lime",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
            elif wellName == "Abanico-19":
                # Draw a rectagle in the plot, semi transparent, from y = 2561.6 till y = 2627.7, that coverets the entire horizontal size of the plot, show the legend and name of each
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2638.8,
                    x1=1,
                    y1=2794.8,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2794.8,
                    x1=1,
                    y1=2972.1,
                    fillcolor="Lime",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
            elif wellName == "Abanico-34":
                # Draw a rectagle in the plot, semi transparent, from y = 2561.6 till y = 2627.7, that coverets the entire horizontal size of the plot, show the legend and name of each
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2782.08,
                    x1=1,
                    y1=2801.25,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2801.25,
                    x1=1,
                    y1=2972.1,
                    fillcolor="Lime",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )

        st.plotly_chart(fig, use_container_width=True)

    with crossPlot:
        plotType = st.radio("Select Plot Type", ["2D", "2D+"])
        if plotType == "2D":
            xAxis = st.selectbox("X-Axis", wellLogList, index=1)
            yAxis = st.selectbox("Y-Axis", wellLogList, index=3)
            fig = px.scatter(df, x=xAxis, y=yAxis)

            st.plotly_chart(fig, use_container_width=True)

        elif plotType == "2D+":
            xAxis = st.selectbox("X-Axis", wellLogList, index=1)
            yAxis = st.selectbox("Y-Axis", wellLogList, index=3)
            colorAxis = st.selectbox("Z-Axis", wellLogList, index=5)
            # colorScale = st.selectbox(
            #     "Color Scale",
            # )
            fig = px.scatter(
                df,
                x=xAxis,
                y=yAxis,
                color=colorAxis,
                color_continuous_scale="Inferno",
            )
            st.plotly_chart(fig, use_container_width=True)


if appMode == "Multi Well Logs":
    # Well 1
    wellName1 = st.sidebar.selectbox(
        "Well X", ("Abanico-2", "Abanico-19", "Abanico-34")
    )
    if wellName1 == "Abanico-2":
        fileName1 = "ABANICO-2_LAS_K.las"
    elif wellName1 == "Abanico-19":
        fileName1 = "ABANICO-19_LAS_K.las"
    elif wellName1 == "Abanico-34":
        fileName1 = "ABANICO-34_LAS_K.las"
    df1 = lasio.read("../data/wells/" + fileName1).df()
    df1 = df1.reset_index()
    wellLogList1 = df1.columns[1:].tolist()

    xAxis = st.sidebar.selectbox("X-Axis", wellLogList1)

    # Well 2
    wellName2 = st.sidebar.selectbox(
        "Well Y", ("Abanico-2", "Abanico-19", "Abanico-34")
    )
    if wellName2 == "Abanico-2":
        fileName2 = "ABANICO-2_LAS_K.las"
    elif wellName2 == "Abanico-19":
        fileName2 = "ABANICO-19_LAS_K.las"
    elif wellName2 == "Abanico-34":
        fileName2 = "ABANICO-34_LAS_K.las"

    df2 = lasio.read("../data/wells/" + fileName2).df()
    df2 = df2.reset_index()
    wellLogList2 = df2.columns[1:].tolist()
    yAxis = st.sidebar.selectbox("Y-Axis", wellLogList2)

    crossPlot = st.tabs(["CrossPlots"])

    xAxis = df1[wellLogList1]
    xAxis = df2[wellLogList2]

    fig = px.scatter(xAxis, yAxis, color_continuous_scale="Inferno")
    st.plotly_chart(fig, use_container_width=True)


if appMode == "Pore Throat Radius":
    st.header = "Pore Throat Radius"
    df = pd.read_csv(
        "../data/K_Phi_Core_RockType_Sorted.csv",
        sep="\t",
        names=headerNames,
    )
    df = df.reset_index()
    # Convert the first column as index
    df = df.set_index("index")
    df_K_Phi = pd.read_csv("../data/K_PHI.csv", sep=",")
    df_R45_P = pd.read_csv("../data/R45-PITTMAN.csv", sep=",")

    # # wellLogList = df.columns[1:].tolist()
    poroSizeSelection = st.sidebar.radio(
        "Select Pore Size",
        ["All", "Nano", "Micro", "Mesos", "Macros", "Megs"],
    )

    if poroSizeSelection == "All":
        df = df
    elif poroSizeSelection == "Nano":
        df = df[df["Rock Type"] == "RT5"]
    elif poroSizeSelection == "Micro":
        df = df[df["Rock Type"] == "RT4"]
    elif poroSizeSelection == "Mesos":
        df = df[df["Rock Type"] == "RT3"]
    elif poroSizeSelection == "Macros":
        df = df[df["Rock Type"] == "RT2"]
    elif poroSizeSelection == "Megs":
        df = df[df["Rock Type"] == "RT1"]

    dataTab, plotTab = st.tabs(["Data", "Plot"])

    with dataTab:
        st.table(df)

    with plotTab:
        # Create a new figure
        fig = go.Figure()

        # Add the first line with red color
        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_10_MEG-MAC"],
                mode="lines",
                name="R_10_MEG-MAC",
                line=dict(color="red"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_2_MAC-MES"],
                mode="lines",
                name="R_2_MEG-MAC",
                line=dict(color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_0_5_MES-MIC"],
                mode="lines",
                name="R_0_5_MES-MIC",
                line=dict(color="lime"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_0_1_MIC-NAN"],
                mode="lines",
                name="R_0_1_MIC-NAN",
                line=dict(color="yellow"),
            )
        )
        # Color-coding for Rock Types
        rock_types = df["Rock Type"].unique()
        colors = [
            "cyan",
            "orange",
            "green",
            "purple",
            "yellow",
        ]  # colors for RT1, RT2, RT3, RT4, and RT5

        # Ensure that the colors list has enough colors for the rock types
        assert len(colors) >= len(rock_types), "Not enough colors for rock types"

        # Add scatter plot for core data
        for rt, color in zip(rock_types, colors):
            mask = df["Rock Type"] == rt
            fig.add_trace(
                go.Scatter(
                    x=df["Porosity 800"][mask] * 100,
                    y=df["K_air 800"][mask],
                    mode="markers",
                    name=f"Core Data - {rt}",
                    marker=dict(color=color),
                )
            )

        # Adjust y-axis to be logarithmic, set its range, and add axis labels
        fig.update_layout(
            yaxis_type="log",
            yaxis_range=[
                -4,
                5,
            ],  # since it's in log scale, this corresponds to your desired range [0.001, 10000]
            xaxis_title="Porosity",
            yaxis_title="Permeability",
        )

        st.plotly_chart(fig, use_container_width=True)

if appMode == "Core Data":
    st.header = "Correlation K-Phi | Rock Type"

    # Create a header with the names of the columns

    df = pd.read_csv(
        "../data/wellCore.csv",
        sep=",",
    )

    statsTab, plotTab, crossPlot, dataTab, histTab = st.tabs(
        ["Stats", "Plot", "CrossPlots", "Data", "Histogram"]
    )

    with statsTab:
        st.table(df.describe())

    with plotTab:
        wellName = st.sidebar.selectbox(
            "Well Name", ("All", "Abanico-2", "Abanico-19", "Abanico-34")
        )
        showLitho = st.radio("Show Lithology", ["Yes", "No"])
        # showRockType = st.radio("Show Rock Type", ["Yes", "No"])

        fig = make_subplots(
            rows=1,
            cols=4,
            shared_yaxes=True,
            subplot_titles=(
                "Porosity",
                "Permeability",
                "R45 Calculated",
                "Poro*Perm",
            ),
        )
        if wellName == "All":
            # From df select only the data from the well Abanico-2
            df = df
        if wellName == "Abanico-2":
            # From df select only the data from the well Abanico-2
            df = df[df["Well Name"] == "Abanico-2"]
        elif wellName == "Abanico-19":
            # From df select only the data from the well Abanico-19
            df = df[df["Well Name"] == "Abanico-19"]
        elif wellName == "Abanico-34":
            # From df select only the data from the well Abanico-34
            df = df[df["Well Name"] == "Abanico-34"]

        fig.add_trace(
            go.Scatter(
                x=df["Porosity"], y=df["Depth"], mode="markers", name="Porosity"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["K_air"], y=df["Depth"], mode="markers", name="K_air"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df["R45"],
                y=df["Depth"],
                mode="markers",
                name="R45",
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=df["PHI*K_air"], y=df["Depth"], mode="markers", name="Poro*Perm"
            ),
            row=1,
            col=4,
        )

        if showLitho == "Yes":
            if wellName == "Abanico-2":
                # Draw a rectagle in the plot, semi transparent, from y = 2561.6 till y = 2627.7, that coverets the entire horizontal size of the plot, show the legend and name of each
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2561.6,
                    x1=1,
                    y1=2627.7,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2627.7,
                    x1=1,
                    y1=2734.7,
                    fillcolor="Lime",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
            elif wellName == "Abanico-19":
                # Draw a rectagle in the plot, semi transparent, from y = 2561.6 till y = 2627.7, that coverets the entire horizontal size of the plot, show the legend and name of each
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2638.8,
                    x1=1,
                    y1=2794.8,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2794.8,
                    x1=1,
                    y1=2972.1,
                    fillcolor="Lime",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
            elif wellName == "Abanico-34":
                # Draw a rectagle in the plot, semi transparent, from y = 2561.6 till y = 2627.7, that coverets the entire horizontal size of the plot, show the legend and name of each
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2782.08,
                    x1=1,
                    y1=2801.25,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=2801.25,
                    x1=1,
                    y1=2972.1,
                    fillcolor="Lime",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    showlegend=True,
                )

                # Invert y axis

        fig.update_yaxes(autorange="reversed")
        fig.update_yaxes(title_text="Depth", row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with crossPlot:
        # xWell = st.sidebar.selectbox("X Well", ["Abanico - 2", "Abanico - 19"])
        # yWell = st.sidebar.selectbox("Y Well", ["Abanico - 2", "Abanico - 19"])

        # xAxis = st.selectbox("X Axis", headerNames, index=1)
        # yAxis = st.selectbox("Y Axis", headerNames, index=2)
        # zAxis = st.selectbox("Z Axis", headerNames, index=4, disabled=True)

        # # fig = px.scatter(x=xWell[xAxis], y=yWell[yAxis])

        # st.plotly_chart(fig, use_container_width=True)
        pass

    with dataTab:
        st.table(df)

    with histTab:
        # histX = st.selectbox("Histogram", headerNames, index=1)
        # fig = px.histogram(df, x=histX, marginal="rug")
        # st.plotly_chart(fig, use_container_width=True)
        pass

if appMode == "Reference Data":
    df_K_Phi = pd.read_csv("../data/K_PHI.csv", sep=",")
    df_R35_W = pd.read_csv("../data/R35_WINLAND.csv", sep=",")
    df_R40_P = pd.read_csv("../data/R40-PITTMAN.csv", sep=",")
    df_R45_P = pd.read_csv("../data/R45-PITTMAN.csv", sep=",")
    df_R50_P = pd.read_csv("../data/R50-PITTMAN.csv", sep=",")

    dataTab, plotTab = st.tabs(["Data", "Plot"])

    with dataTab:
        dataReference = st.sidebar.selectbox(
            "Select Reference Dataset",
            ["K/PHI", "R35-WINLAND", "R40-PITTMAN", "R45-PITTMAN", "R50-PITTMAN"],
        )

        if dataReference == "K/PHI":
            st.write("K/PHI")
            st.table(df_K_Phi)
        elif dataReference == "R35-WINLAND":
            st.write("R35-WINLAND")
            st.table(df_R35_W)
        elif dataReference == "R40-PITTMAN":
            st.write("R40-PITTMAN")
            st.table(df_R40_P)
        elif dataReference == "R45-PITTMAN":
            st.write("R45-PITTMAN")
            st.table(df_R45_P)
        elif dataReference == "R50-PITTMAN":
            st.write("R50-PITTMAN")
            st.table(df_R50_P)

    with plotTab:
        # Create a new figure
        fig = go.Figure()

        # Add the first line with red color
        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_10_MEG-MAC"],
                mode="lines",
                name="R_10_MEG-MAC",
                line=dict(color="red"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_2_MAC-MES"],
                mode="lines",
                name="R_2_MEG-MAC",
                line=dict(color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_0_5_MES-MIC"],
                mode="lines",
                name="R_0_5_MES-MIC",
                line=dict(color="lime"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_K_Phi["PHI"],
                y=df_R45_P["R_0_1_MIC-NAN"],
                mode="lines",
                name="R_0_1_MIC-NAN",
                line=dict(color="yellow"),
            )
        )

        # Adjust y-axis to be logarithmic and set its range
        fig.update_layout(
            yaxis_type="log",
            yaxis_range=[
                0.01,
                4,
            ],  # since it's in log scale, this corresponds to your desired range [0.001, 10000]
        )

        st.plotly_chart(fig, use_container_width=True)
