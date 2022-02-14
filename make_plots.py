import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from bokeh.plotting import figure

def matplotlib_plot(chart_type: str, df,category,chart_x,chart_y,chart_z):
    """ return matplotlib plots """

    Xlabel = chart_x
    Ylabel = chart_y
    Zlabel = chart_z
    fig, ax = plt.subplots()
    if chart_type == "Scatter":
        with st.echo():
            df["color"] = df[category].replace(
                {"Adelie": 1, "Chinstrap": 2, "Gentoo": 3}
            )
            ax.scatter(x=df[Xlabel], y=df[Ylabel], c=df["color"])
            plt.title("Bill Depth by Bill Length")
            plt.xlabel(Xlabel)
            plt.ylabel(Ylabel)
    elif chart_type == "Histogram":
        with st.echo():
            plt.title("Count of Bill Depth Observations")
            ax.hist(df[Xlabel])
            plt.xlabel(Xlabel)
            plt.ylabel("Count")
    elif chart_type == "Bar":
        with st.echo():
            df_plt = df.groupby(category, dropna=False).mean().reset_index()
            ax.bar(x=df_plt[Xlabel], height=df_plt[Ylabel])
            plt.title("Mean Bill Depth by Species")
            plt.xlabel(Xlabel)
            plt.ylabel(Ylabel)

    elif chart_type == "Line":
        with st.echo():
            ax.plot(df.index, df[Ylabel])
            plt.title("Bill Length Over Time")
            plt.ylabel(Ylabel)
    elif chart_type == "3D Scatter":
        ax = fig.add_subplot(projection="3d")
        with st.echo():
            df["color"] = df[category].replace(
                {"Adelie": 1, "Chinstrap": 2, "Gentoo": 3}
            )
            ax.scatter3D(
                xs=df[Xlabel],
                ys=df[Ylabel],
                zs=df[Zlabel],
                c=df["color"],
            )
            ax.set_xlabel(Xlabel)
            ax.set_ylabel(Ylabel)
            ax.set_zlabel(Zlabel)
            plt.title("3D Scatterplot")
    return fig


def sns_plot(chart_type: str, df, category,chart_x,chart_y,chart_z):
    """ return seaborn plots """

    Xlable = chart_x
    Ylable = chart_y
    Zlable = chart_z
    
    fig, ax = plt.subplots()
    if chart_type == "Scatter":
        with st.echo():
            sns.scatterplot(
                data=df,
                x=Xlable,
                y=Ylable,
                hue=category,
            )
            plt.title("Bill Depth by Bill Length")
    elif chart_type == "Histogram":
        with st.echo():
            sns.histplot(data=df, x=Xlable)
            plt.title("Count of Bill Depth Observations")
    elif chart_type == "Bar":
        with st.echo():
            sns.barplot(data=df, x=Xlable, y=Ylable)
            plt.title("Mean Bill Depth by Species")
    elif chart_type == "Boxplot":
        with st.echo():
            sns.boxplot(data=df)
            plt.title("Bill Depth Observations")
    elif chart_type == "Line":
        with st.echo():
            sns.lineplot(data=df, x=df.index, y=Ylabel)
            plt.title("Bill Length Over Time")
    elif chart_type == "3D Scatter":
        st.write("Seaborn doesn't do 3D ☹️. Here's 2D.")
        sns.scatterplot(data=df, x="bill_depth_mm", y="bill_length_mm", hue="island")
        plt.title("Just a 2D Scatterplot")
    return fig


def plotly_plot(chart_type: str, df, category,chart_x,chart_y,chart_z):
    """ return plotly plots """
    
    Xlabel = chart_x
    Ylabel = chart_y
    Zlabel = chart_z
    if chart_type == "Scatter":
        with st.echo():
            fig = px.scatter(
                data_frame=df,
                x= Xlabel,
                y= Ylabel,
                color= category,
                title="Bill Depth by Bill Length",
            )
    elif chart_type == "Histogram":
        with st.echo():
            fig = px.histogram(
                data_frame=df,
                x= Xlable,
                title="Count of Bill Depth Observations",
            )
    elif chart_type == "Bar":
        with st.echo():
            fig = px.histogram(
                data_frame=df,
                x= Xlable,
                y= Ylable,
                title="Mean Bill Depth by Species",
                histfunc="avg",
            )
            # by default shows stacked bar chart (sum) with individual hover values
    elif chart_type == "Boxplot":
        with st.echo():
            fig = px.box(data_frame=df, x= Xlable, y=Ylable)
    elif chart_type == "Line":
        with st.echo():
            fig = px.line(
                data_frame=df,
                x=df.index,
                y= Ylable,
                title="Bill Length Over Time",
            )
    elif chart_type == "3D Scatter":
        with st.echo():
            fig = px.scatter_3d(
                data_frame=df,
                x= Xlable,
                y= Ylable,
                z= Zlable,
                color= category,
                title="Interactive 3D Scatterplot!",
            )

    return fig


def altair_plot(chart_type: str, df, category,chart_x,chart_y,chart_z):
    """ return altair plots """

    Xlable = chart_x
    Ylable = chart_y
    Zlable = chart_z
    if chart_type == "Scatter":
        with st.echo():
            fig = (
                alt.Chart(
                    df,
                    title="Bill Depth by Bill Length",
                )
                .mark_point()
                .encode(x= Xlable, y= Ylable, color= category)
                .interactive()
            )
    elif chart_type == "Histogram":
        with st.echo():
            fig = (
                alt.Chart(df, title="Count of Bill Depth Observations")
                .mark_bar()
                .encode(alt.X(Xlable, bin=True), y="count()")
                .interactive()
            )
    elif chart_type == "Bar":
        with st.echo():
            fig = (
                alt.Chart(
                    df.groupby(category, dropna=False).mean().reset_index(),
                    title="Mean Bill Depth by Species",
                )
                .mark_bar()
                .encode(x= Xlable, y= Ylable)
                .interactive()
            )
    elif chart_type == "Boxplot":
        with st.echo():
            fig = (
                alt.Chart(df).mark_boxplot().encode(x=Xlable, y=Ylable)
            )
    elif chart_type == "Line":
        with st.echo():
            fig = (
                alt.Chart(df.reset_index(), title="Bill Length Over Time")
                .mark_line()
                .encode(x="index:T", y=Ylable)
                .interactive()
            )
    elif chart_type == "3D Scatter":
        st.write("Altair doesn't do 3D ☹️. Here's 2D.")
        fig = (
            alt.Chart(df, title="Just a 2D Scatterplot")
            .mark_point()
            .encode(x="bill_depth_mm", y="bill_length_mm", color="species")
            .interactive()
        )
    return fig


def pd_plot(chart_type: str, df, category,chart_x,chart_y,chart_z):
    """ return pd matplotlib plots """

    Xlable = chart_x
    Ylable = chart_y
    Zlable = chart_z
    fig, ax = plt.subplots()
    if chart_type == "Scatter":
        with st.echo():
            df["color"] = df[category].replace(
                {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}
            )
            ax_save = df.plot(
                kind="scatter",
                x= Xlable,
                y= Ylable,
                c= "color",
                ax=ax,
                title="Bill Depth by Bill Length",
            )
    elif chart_type == "Histogram":
        with st.echo():
            ax_save = df[Xlable].plot(
                kind="hist", ax=ax, title="Count of Bill Depth Observations"
            )
            plt.xlabel(Xlable)
    elif chart_type == "Bar":
        with st.echo():
            ax_save = (
                df.groupby(category, dropna=False)
                .mean()
                .plot(
                    kind="bar",
                    y=Ylable,
                    title="Mean Bill Depth by Species",
                    ax=ax,
                )
            )
            plt.ylabel(Ylable)
    elif chart_type == "Boxplot":
        with st.echo():
            ax_save = df.plot(kind="box", ax=ax)
    elif chart_type == "Line":
        with st.echo():
            ax_save = df.plot(kind="line", use_index=True, y=Ylabel, ax=ax)
            plt.title("Bill Length Over Time")
            plt.ylabel(Ylabel)
    elif chart_type == "3D Scatter":
        st.write("Pandas doesn't do 3D ☹️. Here's 2D.")
        ax_save = df.plot(kind="scatter", x="bill_depth_mm", y="bill_length_mm", ax=ax)
        plt.title("Just a 2D Scatterplot")
    return fig


def bokeh_plot(chart_type: str, df, category,chart_x,chart_y,chart_z):
    """ return bokeh plots """

    Xlabel = chart_x
    Ylabel = chart_y
    Zlabel = chart_z
    if chart_type == "Scatter":
        with st.echo():
            df["color"] = df[category].replace(
                {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}
            )
            fig = figure(title="Bill Depth by Bill Length")
            fig.circle(source=df, x=Xlabel, y=Ylabel, color="color")
    elif chart_type == "Histogram":
        with st.echo():
            hist, edges = np.histogram(df[Xlabel].dropna(), bins=10)
            fig = figure(title="Count of Bill Depth Observations")
            fig.quad(
                top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white"
            )

    elif chart_type == "Bar":
        with st.echo():
            fig = figure(
                title="Mean Bill Depth by Species",
                x_range=["Gentoo", "Chinstrap", "Adelie"],
            )

            fig.vbar(
                source=df.groupby(category, dropna=False).mean(),
                x=Xlabel,
                top=Ylabel,
                width=0.8,
            )

    elif chart_type == "Line":
        with st.echo():
            fig = figure(title="Bill Length Over Time", x_axis_type="datetime")
            fig.line(source=df.reset_index(), x="index", y=Ylabel)

    elif chart_type == "3D Scatter":
        st.write("Bokeh doesn't do 3D ☹️. Here's 2D.")

        df["color"] = df["species"].replace(
            {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}
        )
        fig = figure(title="Bill Depth by Bill Length")
        fig.circle(source=df, x="bill_depth_mm", y="bill_length_mm", color="color")

    return fig
