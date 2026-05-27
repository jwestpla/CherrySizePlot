import gdown
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from docx import Document
import itertools
import re
import matplotlib.pyplot as plt  # For normalization


# --- Step 1: Download the docx from Google Drive ---
file_id = "1EdVCBcDVvejNhifkps9n132LYgLStyjv"
output_path = "metingen.docx"  # Changed from berekeningen.docx
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)


# --- Load the document ---
doc = Document("metingen.docx")  

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
    "Bellise": clean_cherry_table(extract_table("metingen.docx", 0)),
    "Van": clean_cherry_table(extract_table("metingen.docx", 1)),
    "Lapins 2009": clean_cherry_table(extract_table("metingen.docx", 2)),
    "Lapins 2015": clean_cherry_table(extract_table("metingen.docx", 3)),
    "Tamara 2019": clean_cherry_table(extract_table("metingen.docx", 4)),
    "Sweetheart 2009": clean_cherry_table(extract_table("metingen.docx", 5)),
    "Sweetheart 2015": clean_cherry_table(extract_table("metingen.docx", 6)),
}

# --- Build all traces up front ---
fig = go.Figure()
buttons = []
trace_metadata = []

for variety_name, df in varieties.items():
    color_iter = itertools.cycle(px.colors.qualitative.Set2)
    year_col_2026 = next(col for col in df.columns if "2026" in str(col))
    df_2026 = df[["Uke", year_col_2026]].dropna()
    traces = []

    # Compute derivative
    dy = df_2026[year_col_2026].diff()
    dx = df_2026["Uke"].diff()
    derivative = dy / dx

    # Normalize derivative for color scale
    norm = plt.Normalize(vmin=0, vmax=4)  # Expecting range from 0 to 4
    colorscale = px.colors.diverging.RdYlGn  # Red = low, Green = high

    # Add all 2026 colored segments
    num_segments = len(df_2026) - 1
    for i in range(1, len(df_2026)):
        val = derivative.iloc[i]
        color_index = min(int(norm(val) * (len(colorscale) - 1)), len(colorscale) - 1)
        color = colorscale[color_index]
        fig.add_trace(go.Scatter(
            x=df_2026["Uke"].iloc[i-1:i+1],
            y=df_2026[year_col_2026].iloc[i-1:i+1],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=10, color=color),
            showlegend=False,
            visible=False,
            hovertemplate=(
                f"Uke: %{{x}}<br>"
                f"Størrelse: %{{y:.1f}} mm<br>"
                f"Vekstrate: {val:.2f} mm/uke<br>"
                f"---------------"
                "<extra></extra>"
            )
        ))
        traces.append(len(fig.data) - 1)
    # Add all other years
    for col in df.columns[1:]:
        if col == year_col_2026:
            continue
        fig.add_trace(go.Scatter(
            x=df["Uke"],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(color=next(color_iter), width=2),
            marker=dict(size=8, symbol="circle", line=dict(width=1, color='white')),
            hovertemplate=f"Størrelse: %{{y:.1f}} mm<extra>{col}</extra>",
            visible=False,
            showlegend=True
        ))
        traces.append(len(fig.data) - 1)

    # --- Add MEAN only (no std dev) ---
# Drop 2026 and any column that doesn't look like a year
    year_columns = [col for col in df.columns[1:]
                if re.match(r"^\s*20\d{2}", str(col)) and "2026" not in str(col)]

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
        name="Gj.snitt (uten 2026)",
        hovertemplate="Uke: %{x}<br>Gj.snitt: %{y:.1f} mm <extra></extra>",
        line=dict(color='black', width=2, dash='dash'),
        visible=False
    ))
    traces.append(len(fig.data) - 1)

    trace_metadata.append((variety_name, traces, num_segments))

# --- Build dropdown buttons ---
for variety_name, trace_ids, num_segments in trace_metadata:
    visibility = [False] * len(fig.data)

    # Show all 2026 segment traces
    for idx in trace_ids[:num_segments]:
        visibility[idx] = True

    # Set other years and mean to legendonly
    for idx in trace_ids[num_segments:]:
        visibility[idx] = "legendonly"

    buttons.append(dict(
        label=variety_name,
        method="update",
        args=[
            {"visible": visibility}
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
    font=dict(family="Arial", size=18),
    width=1000,
    height=650,
    margin=dict(l=80, r=40, t=80, b=60),
    xaxis=dict(
        title="Ukenummer",
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
        title="Størrelse (mm)",
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
#fig.show()
fig.write_html("cherry_growth_data.html", include_plotlyjs='cdn')
