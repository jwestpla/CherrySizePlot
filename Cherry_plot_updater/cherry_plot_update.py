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
output_path = "metingen.docx"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

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

# --- Load historical tables ---
varieties = {
    "Bellise": clean_cherry_table(extract_table("metingen.docx", 0)),
    "Van": clean_cherry_table(extract_table("metingen.docx", 1)),
    "Lapins 2009": clean_cherry_table(extract_table("metingen.docx", 2)),
    "Lapins 2015": clean_cherry_table(extract_table("metingen.docx", 3)),
    "Tamara 2019": clean_cherry_table(extract_table("metingen.docx", 4)),
    "Sweetheart 2009": clean_cherry_table(extract_table("metingen.docx", 5)),
    "Sweetheart 2015": clean_cherry_table(extract_table("metingen.docx", 6)),
}

# --- Extract and Parse the separate 2026 Table ---
# According to your doc, Table 7 is "Tilvekst tabell 2025" and Table 8 is "Tilvekst tabell 2026"
df_2026_raw = extract_table("metingen.docx", 8)
df_2026_raw.columns = ["Variety"] + df_2026_raw.iloc[0, 1:].tolist()
df_2026_raw = df_2026_raw.iloc[1:]

# Map document names to your dictionary keys
name_mapping = {
    "Bellise": "Bellise",
    "Van": "Van",
    "Lapins 09": "Lapins 2009",
    "Lapins 15": "Lapins 2015",
    "Tamara": "Tamara 2019",
    "Sweetheart 09": "Sweetheart 2009",
    "Sweetheart 15": "Sweetheart 2015"
}

# Inject the 2026 data column into the historical variety dataframes
for doc_name, dict_key in name_mapping.items():
    # Find row for this variety in the 2026 growth rate table
    variety_row = df_2026_raw[df_2026_raw["Variety"].str.contains(doc_name, na=False, case=False)]
    if not variety_row.empty:
        # Build a temporary dataframe for the 2026 sizes
        weeks_2026 = []
        sizes_2026 = []
        for col in df_2026_raw.columns[1:]:
            wk_num = int(re.search(r'\d+', col).group())
            val = variety_row[col].values[0].replace(',', '.')
            weeks_2026.append(wk_num)
            sizes_2026.append(pd.to_numeric(val, errors='coerce'))
        
        df_new = pd.DataFrame({"Uke": weeks_2026, "2026": sizes_2026})
        
        # Merge the new 2026 column back into the historical dataframe
        varieties[dict_key] = pd.merge(varieties[dict_key], df_new, on="Uke", how="outer")

# --- Build all traces up front ---
fig = go.Figure()
buttons = []
trace_metadata = []

for variety_name, df in varieties.items():
    color_iter = itertools.cycle(px.colors.qualitative.Set2)
    
    # Check if 2026 exists after merge, otherwise fall back to 2025 safely
    if "2026" in df.columns:
        year_col = "2026"
    else:
        year_col = "2025"
        
    df_current = df[["Uke", year_col]].dropna()
    traces = []

    # Compute derivative
    num_segments = 0
    if len(df_current) > 1:
        dy = df_current[year_col].diff()
        dx = df_current["Uke"].diff()
        derivative = dy / dx

        norm = plt.Normalize(vmin=0, vmax=4)
        colorscale = px.colors.diverging.RdYlGn

        num_segments = len(df_current) - 1
        for i in range(1, len(df_current)):
            val = derivative.iloc[i]
            if pd.isna(val):
                val = 0
            color_index = min(int(norm(val) * (len(colorscale) - 1)), len(colorscale) - 1)
            color = colorscale[color_index]
            fig.add_trace(go.Scatter(
                x=df_current["Uke"].iloc[i-1:i+1],
                y=df_current[year_col].iloc[i-1:i+1],
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

    # Add all other background years
    for col in df.columns[1:]:
        if col == year_col:
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

    # Add historical mean baseline line
    year_columns = [col for col in df.columns[1:] if re.match(r"^\s*20\d{2}", str(col)) and col != year_col]
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
        name=f"Gj.snitt (uten {year_col})",
        hovertemplate="Uke: %{x}<br>Gj.snitt: %{y:.1f} mm <extra></extra>",
        line=dict(color='black', width=2, dash='dash'),
        visible=False
    ))
    traces.append(len(fig.data) - 1)

    trace_metadata.append((variety_name, traces, num_segments))

# --- Build dropdown buttons ---
for variety_name, trace_ids, num_segments in trace_metadata:
    visibility = [False] * len(fig.data)
    for idx in trace_ids[:num_segments]:
        visibility[idx] = True
    for idx in trace_ids[num_segments:]:
        visibility[idx] = "legendonly"

    buttons.append(dict(
        label=variety_name,
        method="update",
        args=[{"visible": visibility}]
    ))

# --- Layout configuration ---
fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=1.02, xanchor="left", y=1, yanchor="top"
    )],
    font=dict(family="Arial", size=18),
    width=1000, height=650,
    margin=dict(l=80, r=40, t=80, b=60),
    xaxis=dict(title="Ukenummer", showline=True, linewidth=2, linecolor='black', ticks="outside", dtick=1, tickmode='linear', tickformat='d'),
    yaxis=dict(title="Størrelse (mm)", range=[12, 31], dtick=1, showgrid=True, gridcolor='lightgray', showline=True, linewidth=2, linecolor='black', ticks="outside"),
    legend_title="Year",
    hovermode="x unified",
    hoverlabel=dict(font_size=14, namelength=0),
    transition=dict(duration=300, easing="cubic-in-out")
)

fig.write_html("cherry_growth_data.html", include_plotlyjs='cdn')
