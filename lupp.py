
import argparse
from collections import defaultdict
from itertools import combinations

import pysam
import numpy as np
import plotly.express as px
from tqdm import tqdm, trange

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

# TODO 
#  - update style e.g bootstrap (https://dash-bootstrap-components.opensource.faculty.ai/docs/)
#  - precompute larger area surrounding region/window, enables panning 
#  - precompute all matrixes for faster switching


app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'info': '#CCF1FF'
}

app.layout = html.Div([
    html.H1(
        children="LUPP", 
        style={
            'textAlign': 'center',
            'color': colors['text'],
            "font-family": "Garamond"
        }
    ),
    html.Div([
        html.Div(children=[
            dcc.Graph(id='plot', style={'height': '85vh'}), 
        ], style={'padding': 10, 'flex': 2}),
        html.Div(children=[
            html.Label('Barcoded BAM', style={'color': colors['text']}),
            html.Br(),
            dcc.Input(
                id="input_bam", type="text", placeholder="Input path to barcoded BAM (tag 'BX')", required=True, 
                style={'padding': 5, 'margin': 5, "width": "300px"}
            ),
            html.Br(),
            html.Label('Region', style={'color': colors['text']}),
            html.Br(),
            dcc.Input
                (id="region", type="text", placeholder="Region to check (e.g chrX 10010 12000)", required=True, 
                style={'padding': 5, 'margin': 5, "width": "300px"}
            ),
            html.Br(),
            html.Label('Expansion factor', style={'color': colors['text']}),
            html.Br(),
            dcc.Slider(
                id='expansion',
                min=1,
                max=10,
                step=0.1,
                value=1,
                marks={i: '{:}x'.format(i) for i in range(1,11)},
            ),
            html.Br(),
            html.Label('Resolution (bp)', style={'color': colors['text']}),
            html.Br(),
            dcc.Input(
                id='resolution', type='number', value=1000, placeholder="Bin size in bp", min=10, max=10000, step=10, 
                style={'padding': 5, 'margin': 5, "width": "100px"}
            ),
            html.Br(),
            html.Button(
                'GENERATE MAP', id='button', n_clicks=0, 
                style={'padding': 10, 'margin': 5}),
            dcc.Checklist(
                id="checks",
                options=[
                    {'label': 'HP map', 'value': "HP"},
                    {'label': 'Normalize', 'value': "Normalize"},
                ],
                style={'color': colors['text'], 'padding': 5},
                value=["Normalize"],
            ),
            html.Hr(),
            html.Pre(id='text', style={'color': colors['info']}),
        ], style={'padding': 10, 'flex': 1, 'width': '30vh'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
], style={'backgroundColor': colors['background'], "justify-content": "center"})


@app.callback(Output('plot', 'figure'),
              Output('text', 'children'),
              Input('button', 'n_clicks'), 
              State('checks', 'value'),
              State('input_bam', 'value'),
              State('region', 'value'),
              State('expansion', 'value'),
              State('resolution', 'value'),
              prevent_initial_call=True)
def main(n_clicks, mode, input_bam, region, expand, resolution):
    if input_bam is None:
        return {}, "Need BAM to be set."

    if region is None:
        return {}, "Need 'region' to be set."

    text = f"Input region: {repr(region)}\n"
    region = region.split(" ")
    bins = []
    with pysam.AlignmentFile(input_bam, mode="rb") as f:
        window = region.copy()

        # Expand region and get the constraints of the window
        buffer = int((int(window[2]) - int(window[1])) * expand)
        window[1] = max(resolution * (int(window[1]) // resolution) - buffer, 0)
        max_len = f.header.get_reference_length(window[0])
        window[2] = min(resolution * (int(window[2]) // resolution) + buffer, max_len)

        text += f"Window used: {repr(window)}\n"

        nr_bins = (window[2] - window[1]) // resolution
        text += f"Nr bins: {nr_bins}\n"

        if nr_bins < 20:
            text += f"ALERT: Low bin count! Increase resolution"

        if nr_bins > 1_000:
            text += f"WARNING: High bin count ({nr_bins})! Decrease resolution"
            return {}, text
        
        parser = f.fetch(until_eof=True) if window is None else f.fetch(*window, until_eof=True)

        if "HP" in mode:
            matrix = get_haplo_matrix(parser, bins, window, resolution)
        else:
            matrix = get_barcode_matrix(parser, bins, window, resolution)
    
    if len(matrix) < nr_bins:
        text += "WARNING: No barcoded reads in region\n"
        matrix = np.zeros((nr_bins, nr_bins))
        bins = range(window[1], window[2]-resolution, resolution)
        print(len(matrix), len(bins))

    # Normalize matrix
    if "Normalize" in mode:
        matrix = np.log1p(matrix)
        norm = HiCNorm(matrix)
        norm.iterative_correction(max_iter=500)
        norm_matrix = norm.norm_matrix
    else:
        norm_matrix = matrix
    
    bin_labels = [f'{b:,}' for b in bins]

    bin_array = np.array(bins)
    start = np.arange(len(bins))[np.argmin(np.abs(bin_array - int(region[1])))]-0.5
    end = np.arange(len(bins))[np.argmin(np.abs(bin_array - int(region[2])))]+0.5

    # Print heatmap
    color = "Reds" if not "HP" in mode else "Purples"
    fig = px.imshow(norm_matrix, color_continuous_scale=color, origin='lower', x=bin_labels, y=bin_labels)
    fig.update_xaxes(tick0=bins[0]-0.5, dtick=1, tickmode="auto", nticks=10, tickangle=-90)
    fig.update_yaxes(tick0=bins[0]-0.5, dtick=1, tickmode="auto", nticks=10)

    # Add lines for highligthing region
    fig.add_vline(
        x=start, line_width=2, line_color="grey")
    fig.add_vline(
        x=end, line_width=2, line_color="grey")
    fig.add_hline(
        y=start, line_width=2, line_color="grey")
    fig.add_hline(
        y=end, line_width=2, line_color="grey")
    return fig, text

def get_barcode_matrix(parser, bins, window, resolution):
    bin_counts = defaultdict(set)
    
    current_bin = window[1]
    bins.append(current_bin)

    for alignment in tqdm(parser, desc="Parsing window"):
        barcode = get_tag(alignment, "BX")
        if barcode is None:
            continue

        if alignment.reference_start - current_bin > resolution:
            current_bin += resolution
            bins.append(current_bin)

        bin_counts[current_bin].add(barcode)

    bin_index = dict(zip(bins, range(len(bins))))
    matrix = np.zeros((len(bins), len(bins)))
    total_combos = (len(bins) * (len(bins) - 1)) // 2
    for bin1, bin2 in tqdm(combinations(bins, 2), desc="Computing matrix", total=total_combos):
        bcs1 = bin_counts[bin1]
        bcs2 = bin_counts[bin2]
        i1 = bin_index[bin1]
        i2 = bin_index[bin2]
        common = len(bcs1 & bcs2)
        matrix[i1, i2] = common
        matrix[i2, i1] = common
    
    return matrix


def get_haplo_matrix(parser, bins, window, resolution):
    bin_counts1 = defaultdict(set)
    bin_counts2 = defaultdict(set)
    
    current_bin = window[1]
    bins.append(current_bin)

    for alignment in tqdm(parser, desc="Parsing window"):
        barcode = get_tag(alignment, "BX")
        if barcode is None:
            continue

        hp = get_tag(alignment, "HP")
        if hp is None:
            continue

        if alignment.reference_start - current_bin > resolution:
            current_bin += resolution
            bins.append(current_bin)

        if hp == 1:
            bin_counts1[current_bin].add(barcode)
        elif hp == 2:
            bin_counts2[current_bin].add(barcode)
        else:
            print(f"Unknown HP={hp}")

    bin_index = dict(zip(bins, range(len(bins))))
    matrix = np.zeros((len(bins), len(bins)))
    total_combos = (len(bins) * (len(bins) - 1)) // 2
    for bin1, bin2 in tqdm(combinations(bins, 2), desc="Computing matrix", total=total_combos):
        bcs_hp1_1 = bin_counts1[bin1]
        bcs_hp1_2 = bin_counts1[bin2]
        bcs_hp2_1 = bin_counts2[bin1]
        bcs_hp2_2 = bin_counts2[bin2]
        i1 = bin_index[bin1]
        i2 = bin_index[bin2]
        common_hp1 = len(bcs_hp1_1 & bcs_hp1_2)
        common_hp2 = len(bcs_hp2_1 & bcs_hp2_2)
        matrix[i1, i2] = common_hp1
        matrix[i2, i1] = common_hp2
    
    return matrix


class HiCNorm(object):
    """
    Contact Matrix normalization
    Taken from https://dearxxj.github.io/post/6/
    """
    def __init__(self, matrix):
        self.bias = None
        self.matrix = matrix
        self.norm_matrix = None
        self._bias_var = None

    def iterative_correction(self, max_iter=100):
        mat = np.array(self.matrix, dtype=float)  # use float to avoid int overflow
        # remove low count rows for normalization
        row_sum = np.sum(mat, axis=1)
        low_count = np.quantile(row_sum, 0.15)  # 0.15 gives best correlation between KR/SP/VC
        mask_row = row_sum < low_count
        mat[mask_row, :] = 0
        mat[:, mask_row] = 0

        self.bias = np.ones(mat.shape[0])
        self._bias_var = []
        x, y = np.nonzero(mat)
        mat_sum = np.sum(mat)   # force the sum of matrix to stay the same, otherwise ICE may not converge as fast
        delta = None
        for i in trange(max_iter, desc="Correcting matrix"):
            bias = np.sum(mat, axis=1)
            bias_mean = np.mean(bias[bias > 0])
            bias_var = np.var(bias[bias > 0])
            self._bias_var.append(bias_var)
            bias = bias / bias_mean
            bias[bias == 0] = 1

            mat[x, y] = mat[x, y] / (bias[x]*bias[y])
            new_sum = np.sum(mat)
            mat = mat * (mat_sum / new_sum)

            # update total bias
            self.bias = self.bias * bias * np.sqrt(new_sum / mat_sum)

            delta = abs(self._bias_var[-1] - self._bias_var[-2]) if i > 1 else 1
            # print(f"Iter {i}: {delta}")
            if delta < 0.0001:
                print(f"Normalization converged after {i} iterations.")
                break

        self.norm_matrix = np.array(self.matrix, dtype=float)
        self.norm_matrix[x, y] = self.norm_matrix[x, y] / (self.bias[x] * self.bias[y])


def get_tag(aln, tag):
    try:
        return aln.get_tag(tag)
    except KeyError:
        return None


if __name__ == '__main__':
    app.run_server(debug=False)
