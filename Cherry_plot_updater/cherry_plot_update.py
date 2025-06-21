import gdown
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from docx import Document
import itertools
import re


# --- Step 1: Download the docx from Google Drive ---
file_id = "1EdVCBcDVvejNhifkps9n132LYgLStyjv"
output_path = "berekeningen.docx"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)


# --- Load the document ---
doc = Document("berekeningen.docx")

# --- Utility functions ---
def extract_table(doc_path, index):
    doc = Document(doc_path)
    table = doc.tables[index]
    data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    return pd.DataFrame(data)

def clean_cherry_table(df_raw):
    df = df_raw.copy()
    df.columns = ["Uke"] + df.iloc[0, 1:].tolist()
    df = df.iloc[1:]
    df["Uke"] = df["Uke"].str.extract(r'(\d+)').astype(int)
    for col in df.columns[1:]:
        df[col] = df[col].replace('', np.nan)
        df[col] = df[col].str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# --- Load and clean tables ---
varieties = {
    "Bellise": clean_cherry_table(extract_table("berekeningen.docx", 0)),
    "Van": clean_cherry_table(extract_table("berekeningen.docx", 1)),
    "Lapins 2009": clean_cherry_table(extract_table("berekeningen.docx", 2)),
    "Lapins 2015": clean_cherry_table(extract_table("berekeningen.docx", 3)),
    "Tamara 2019": clean_cherry_table(extract_table("berekeningen.docx", 4)),
    "Sweetheart 2009": clean_cherry_table(extract_table("berekeningen.docx", 5)),
    "Sweetheart 2015": clean_cherry_table(extract_table("berekeningen.docx", 6)),
}

# --- Build all traces up front ---
fig = go.Figure()
buttons = []
trace_metadata = []

for variety_name, df in varieties.items():
    color_iter = itertools.cycle(px.colors.qualitative.Set2)
    year_col_2025 = next(col for col in df.columns if "2025" in str(col))
    df_2025 = df[["Uke", year_col_2025]].dropna()
    traces = []

    # Add 2025 trace
    fig.add_trace(go.Scatter(
        x=df_2025["Uke"],
        y=df_2025[year_col_2025],
        mode='lines+markers',
        name="2025",
        line=dict(color='black', width=4),
        marker=dict(size=10, symbol="circle", line=dict(width=2, color='white')),
        hovertemplate="Size: %{y:.1f} mm<extra>2025</extra>",
        visible=False,
        showlegend=True
    ))
    traces.append(len(fig.data) - 1)
    # TEST
    # Add all other years
    for col in df.columns[1:]:
        if col == year_col_2025:
            continue
        fig.add_trace(go.Scatter(
            x=df["Uke"],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(color=next(color_iter), width=2),
            marker=dict(size=8, symbol="circle", line=dict(width=1, color='white')),
            hovertemplate=f"Size: %{{y:.1f}} mm<extra>{col}</extra>",
            visible=False,
            showlegend=True
        ))
        traces.append(len(fig.data) - 1)

    # --- Add MEAN only (no std dev) ---
# Drop 2025 and any column that doesn't look like a year
    year_columns = [col for col in df.columns[1:]
                if re.match(r"^\s*20\d{2}", str(col)) and "2025" not in str(col)]

    df_subset = df[year_columns]

    valid_counts = df_subset.count(axis=1)
    mean_series = df_subset.mean(axis=1).where(valid_counts >= 2)

    valid_rows = mean_series.notna()
    weeks = df["Uke"][valid_rows]
    mean_vals = mean_series[valid_rows]

    fig.add_trace(go.Scatter(
        x=weeks,
        y=mean_vals,
        mode='lines',
        name="Mean (excl. 2025)",
        line=dict(color='black', width=2, dash='dash'),
        visible=False
    ))
    traces.append(len(fig.data) - 1)

    trace_metadata.append((variety_name, traces))

# --- Build dropdown buttons ---
for variety_name, trace_ids in trace_metadata:
    visibility = [False] * len(fig.data)
    visibility[trace_ids[0]] = True  # show 2025
    for idx in trace_ids[1:]:        # keep rest in legend only
        visibility[idx] = "legendonly"

    buttons.append(dict(
        label=variety_name,
        method="update",
        args=[
            {"visible": visibility},
            {"title": f"{variety_name} (2015): Growth by Year"}
        ]
    ))

# --- Layout configuration ---
fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=1.02,
        xanchor="left",
        y=1,
        yanchor="top"
    )],
    title="Bellise (2015): Growth by Year",
    font=dict(family="Arial", size=18),
    width=1000,
    height=650,
    margin=dict(l=80, r=40, t=80, b=60),
    xaxis=dict(
        title="Week Number",
        showline=True,
        linewidth=2,
        linecolor='black',
        ticks="outside",
        tickwidth=2,
        ticklen=8,
        mirror=True,
        dtick=1,
        tickmode='linear',
        tickformat='d'
    ),
    yaxis=dict(
        title="Size (mm)",
        range=[12, 31],
        dtick=1,
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor='black',
        ticks="outside",
        tickwidth=2,
        ticklen=8,
        mirror=True
    ),
    legend_title="Year",
    hovermode="x unified",
    hoverlabel=dict(font_size=14, namelength=0),
    transition=dict(duration=300, easing="cubic-in-out")
)

# --- Show or export ---
fig.show()
# fig.write_html("cherry_growth_plot.html", include_plotlyjs='cdn')