import numpy as np 
import pandas as pd
import dash
from dash import Dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
from seaborn import heatmap
import plotly.graph_objects as go
import random as rd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
tabs_styles = {'zIndex': 99, 'display': 'inlineBlock', 'height': '4vh', 'width': '12vw',
               'position': 'fixed', "background": "#323130", 'top': '12.5vh', 'left': '7.5vw',
               'border': 'grey', 'border-radius': '4px'}
tab_style = {
    "background": "black",
    'text-transform': 'uppercase',
    'color': '#54FF9F',
    'border': '#147852',
    'font-size': '11px',
    'font-weight': 600,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '4px',
    'padding':'6px'
}


datainit = pd.read_csv('transcriptomics_data.csv')   
datawithout = datainit.drop(columns=["cell_type", "colour"])
datatranspose = datawithout.T
datatranspose['average'] = datatranspose.mean(axis=1)

datarfg = datainit.copy()

def remove_flat_genes(db):
    ret = db.copy()
    for col in db:
        if col == 'cell_type' or col == 'colour':
            continue

        q25 = db[col].quantile(.25)
        q75 = db[col].quantile(.75)
        if q25 == q75:
            ret.drop(col, axis=1, inplace=True)
        
        """medi = db[col].median()
        mini = db[col].min()
        if medi == mini:
            ret.drop(col, axis=1, inplace=True)"""

    return ret

datarfg = remove_flat_genes(datainit)
"""
datarfgwithout = datarfg.drop(columns=["cell_type", "colour"])
tsne = TSNE(n_components=2, random_state=1)
results_tsne_w = tsne.fit_transform(datawithout)
results_tsne_rfg = tsne.fit_transform(datarfgwithout)"""

    

app.layout = html.Div([
    html.H1('Single-cell RNA transcriptomics visualisation', style ={'color' : '#54FF9F', 'text-align' : 'center'}),
    dcc.Tabs([
        dcc.Tab(label = 'Cells Overview', style=tab_style, selected_style=tab_style, children=[
            html.Div([
                dcc.Graph(id = 'cell_traceplot'),
                dcc.Checklist(
                id="rfg_c",
                options=["Remove flat genes"],
                value=[]),
                dcc.Checklist(
                id="sort_genes_c",
                options=["Sort genes"],
                value=[]),
                dcc.Checklist(
                id="show_mean",
                options=["Show mean"],
                value=[])
            ]),
            html.Div([
                html.Br(),
                dcc.Input(id = "cell_traceplot_number",
                    type="number",
                    placeholder="cell to display",
                    min = 0, 
                    max = 23822,
                    style={'backgroundColor': 'white', 'color': 'black'},
                    value=0
                )   
            ])
        ]),
        dcc.Tab(label = 'Cell Types Overview', style=tab_style, selected_style=tab_style, children=[
            html.Div([
                dcc.Graph(id = 'groups_boxplot'),
                dcc.Checklist(
                id="rfg",
                options=["Remove flat genes"],
                value=[]),
                dcc.Checklist(
                id="sort_genes",
                options=["Sort genes"],
                value=[])
            ]),
            html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id = "group_boxplot_group_to_display",
                            placeholder="group to display", 
                            options=sorted([{'label': i, 'value': i} for i in datainit.cell_type.unique()], key = lambda x: x['label']),
                            style={'backgroundColor': 'white', 'color': 'black'}, 
                            value = 0,
                            multi = False)),
                    
                    dbc.Col(
                        dcc.Dropdown(
                            id = "group_boxplot_gene_to_display",
                            placeholder="gene to display", 
                            options=[{'label': i, 'value': i} for i in datawithout.columns],
                            style={'backgroundColor': 'white', 'color': 'black'},
                            value = ['all'],
                            multi = True
                        ))
                    ], justify = 'center', align = 'center', className = 'display box')
            ]),
            html.Div([
                dcc.Graph(id='correlation_cell_types'),
                dcc.Checklist(
                id="rfg_corr",
                options=["Remove flat genes"],
                value=[])
            ]),
            html.Div([
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id = "corr_first_cell_type",
                            placeholder="first cell type to display", 
                            options=sorted([{'label': i, 'value': i} for i in datainit.cell_type.unique()], key = lambda x: x['label']),
                            style={'backgroundColor': 'white', 'color': 'black'}, 
                            value = 0,
                            multi = False
                    )),                    
                    dbc.Col(
                        dcc.Dropdown(
                            id = "corr_second_cell_type",
                            placeholder="second cell type to display", 
                            options=sorted([{'label': i, 'value': i} for i in datainit.cell_type.unique()], key = lambda x: x['label']),
                            style={'backgroundColor': 'white', 'color': 'black'},
                            value = 1,
                            multi = False
                        ))
                    ], justify = 'center', align = 'center', className = 'display box')
            ]),
            html.Div([
                dcc.Graph(id='heatmap_cell_types'),
                dcc.Checklist(
                id="rfg_heatmap",
                options=["Remove flat genes"],
                value=[],)
            ])
        ]),
        dcc.Tab(label = 'Clustering and Dimensionality Reduction',  style=tab_style, selected_style=tab_style, children=[
            html.Div([
                dcc.RadioItems(
                id="clust_and_dr_content",
                options=["Genes", "Cell types", "Cells (performance hazard)"],
                inline=True,
                value="Genes"),
            ],style={"display": "flex", "justifyContent": "center"}),
            html.Div([
                dcc.Graph(id = 'clustering_cell_types_2'),
                dcc.Checklist(
                id="rfg_clust_2",
                options=["Remove flat genes"],
                value=[]),
                dcc.RadioItems(
                id="clustering_type_2",
                options=["None", "K-Means", "Hierarchical Clustering", "Density-based Clustering"],
                inline=True,
                value='None'),
                dcc.Input(id = "number_of_clusters_2",
                    type="number",
                    placeholder="Number of clusters",
                    min = 0, 
                    max = 1000,
                    style={'backgroundColor': 'white', 'color': 'black'},
                    value=3
                ),
                dcc.Input(id = "db_eps_2",
                    type="number",
                    placeholder="Epsilon value",
                    min = 0, 
                    max = 100,
                    style={'backgroundColor': 'white', 'color': 'black'},
                    value=1
                ),
                dcc.Input(id = "db_min_samples_2",
                    type="number",
                    placeholder="Minimum samples",
                    min = 0, 
                    max = 100,
                    style={'backgroundColor': 'white', 'color': 'black'},
                    value=8
                )  
            ]),
            html.Div([
                dcc.Graph(id = 'tsne_graph'),
                dcc.Checklist(
                id="rfg_tsne",
                options=["Remove flat genes"],
                value=[]),
                dcc.RadioItems(
                id="clustering_tsne",
                options=["None", "K-Means", "Hierarchical Clustering", "Density-based Clustering"],
                inline=True,
                value="None"),
                dcc.Input(id = "number_of_clusters_tsne",
                    type="number",
                    placeholder="Number of clusters",
                    min = 0, 
                    max = 1000,
                    style={'backgroundColor': 'white', 'color': 'black'},
                    value=3
                ),
                dcc.Input(id = "db_eps_tsne",
                    type="number",
                    placeholder="Epsilon value",
                    min = 0, 
                    max = 100,
                    style={'backgroundColor': 'white', 'color': 'black'},
                    value=1
                ),
                dcc.Input(id = "db_min_samples_tsne",
                    type="number",
                    placeholder="Minimum samples",
                    min = 0, 
                    max = 100,
                    style={'backgroundColor': 'white', 'color': 'black'},
                    value=8
                )  
            ]),
            
        ])
        
    ])
])


@app.callback(
    Output('cell_traceplot', 'figure'),
    Input('cell_traceplot_number', 'value'),
    Input('rfg_c', 'value'),
    Input('sort_genes_c', 'value'),
    Input('show_mean', 'value')) 
def update_cells_overview(selected_cell, rfg, srt, show_mean):
    if selected_cell == None:
        selected_cell = 0
    if rfg:
        db = datarfg
    else:
        db = datainit
    data = db.drop(columns=["cell_type", "colour"])
    data = data.loc[selected_cell]
    if srt:
        data = data.sort_values()
    fig = px.line(data, template="plotly_dark", color_discrete_map={"Average": "#456987", selected_cell: "#147852"}, title="Traceplot for each cell")
    if show_mean:
        mn = data.mean()
        fig.add_hline(mn)
    return fig

@app.callback(
    Output("group_boxplot_gene_to_display", 'options'),
    Input("rfg", "value")
)
def update_group_overview_dropdown_1(rfg):
    if rfg:
        temp = datarfg.drop(["cell_type", "colour"], axis=1)
        return [{'label': 'Select all', 'value': 'all'}] + [{'label': i, 'value': i} for i in temp.columns]
    else:
        return [{'label': 'Select all', 'value': 'all'}] + [{'label': i, 'value': i} for i in datawithout.columns]

@app.callback(
    Output('groups_boxplot', 'figure'),
    Input('group_boxplot_group_to_display', 'value'), 
    Input('group_boxplot_gene_to_display', 'value'),
    Input('rfg', 'value'),
    Input('sort_genes', 'value'))
def update_group_overview_graph_1(selected_group, selected_gene, rfg, srt):

    fig1 = go.Figure(layout=dict(template="plotly_dark"))

    if type(selected_gene) is str:
        selected_gene = [selected_gene]
        """db = datainit.copy()
        data = db[db.cell_type == selected_group].loc[:,selected_gene]    
        xdata = selected_gene
        ydata = []
        ydata.append(data.tolist())"""
    else:
        if rfg:
            db = datarfg
        else:
            db = datainit
        data = db
        if 'all' in selected_gene : 
            data = data[data.cell_type == selected_group]
            data = data.drop(columns=["cell_type", "colour"])
        else:
            data = data[data.cell_type == selected_group].loc[:,selected_gene]
        if srt:
            data = data.T
            data['median'] = data.median(axis=1)
            data.sort_values(by=['median'], inplace=True)
            data = data.T
            data = data.drop('median')
        genes = data.columns
        
    for gene in genes:
        values = data.loc[:,gene]
        fig1.add_trace(go.Box(
            y=values,
            name=gene,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=1))
        
    fig1.update_layout(
        title='Ranges of values of each gene for a cell type',
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            dtick=5,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2
        ),
        margin=dict(
            l=40,
            r=30,
            b=80,
            t=100,
        ),
        showlegend=False,
    )   
    return fig1

@app.callback(
    Output('correlation_cell_types', 'figure'),
    Input('corr_first_cell_type', 'value'),
    Input('corr_second_cell_type', 'value'),
    Input('rfg_corr', 'value')
)
def correlation_cell_types(first_ct, sec_ct, rfg):
    fig = go.Figure(layout=dict(template="plotly_dark"), layout_yaxis_range=[-14,14], layout_xaxis_range=[-14,14])
    fig.update_layout(title='Cell type correlation')
    if rfg:
        db = datarfg
    else:
        db = datainit

    frst = db[db.cell_type == first_ct]
    secnd = db[db.cell_type == sec_ct]
    
    colors = '#54FF9F'
    frst = frst.drop(columns=["cell_type", "colour"])
    secnd = secnd.drop(columns=["cell_type", "colour"])

    f = []
    for col in frst:
        f.append(frst[col].mean())
    s = []
    for col in secnd:
        s.append(secnd[col].mean())
    
    fig.add_trace(go.Scatter(x=f, y=s, mode='markers',marker=dict(color=colors)))
    return fig

@app.callback(
    Output('clustering_cell_types_2', 'figure'),
    Input('rfg_clust_2', 'value'),
    Input('clustering_type_2', 'value'),
    Input('number_of_clusters_2', 'value'),
    Input('db_eps_2', 'value'),
    Input('db_min_samples_2', 'value'),
    Input('clust_and_dr_content', 'value')
)
def dr_and_clustering_cells(rfg, clust_type, K, eps, ms, content):
    fig = go.Figure(layout=dict(template="plotly_dark"))
    
    if rfg:
        db = datarfg
    else:
        db = datainit

    colors = '#54FF9F'

    if content == "Genes":
        fig.update_layout(title=f'PCA on each gene')
        data = db.drop(columns=['cell_type', 'colour'])
        names = data.columns
        names = names.T
        data = data.T
        pca = PCA(n_components=2)
    elif content == "Cell types":
        fig.update_layout(title=f'PCA on each cell type')
        ct_df = []
        colors = []
        names = []
        for i in range(133):
            temp = []
            names.append(f"cell type {i}")
            cur_ct_df = db[db.cell_type == i].drop(columns=['cell_type'])
            for col in cur_ct_df:
                if col == 'colour':
                    colors.append(cur_ct_df[col].iloc[0])
                else:
                    temp.append(cur_ct_df[col].mean())
            ct_df.append(temp)
        data = pd.DataFrame(ct_df)
        pca = PCA(n_components=2)
    elif content == "Cells (performance hazard)":
        fig.update_layout(title=f'PCA on each cell')
        colors = db['colour']
        names = list(range(1, 23823))
        data = db.drop(columns=['cell_type', 'colour'])
        pca = PCA(n_components=2)

    
    results_pca = pca.fit_transform(data)
    X_pca = []
    Y_pca = []
    for pt in results_pca:
        X_pca.append(pt[0])
        Y_pca.append(pt[1])

    if clust_type == "None":
        colors = colors
    elif clust_type == "K-Means":
        colors = kmeans_clustering(X_pca, Y_pca, K)
    elif clust_type == "Hierarchical Clustering":
        colors = hierarchical_clustering(X_pca, Y_pca, K)
    elif clust_type == "Density-based Clustering":
        colors = densitybased_clustering(X_pca, Y_pca, eps, ms)

    fig.add_trace(go.Scatter(x=X_pca, y=Y_pca, mode='markers',marker=dict(color=colors), text=names))
    return fig
   
@app.callback(
    Output('tsne_graph', 'figure'),
    Input('rfg_tsne', 'value'),
    Input('clustering_tsne', 'value'),
    Input('number_of_clusters_tsne', 'value'),
    Input('db_eps_tsne', 'value'),
    Input('db_min_samples_tsne', 'value'),
    Input('clust_and_dr_content', 'value')
)
def tsne_cells(rfg, clust_type, K, eps, ms, content):
    fig = go.Figure(layout=dict(template="plotly_dark"))
    
    if rfg:
        db = datarfg
    else:
        db = datainit

    colors = '#54FF9F'

    if content == "Genes":
        fig.update_layout(title=f't-SNE on each gene')
        data = db.drop(columns=['cell_type', 'colour'])
        names = data.columns
        names = names.T
        data = data.T
    elif content == "Cell types":
        fig.update_layout(title=f't-SNE on each cell type')
        ct_df = []
        colors = []
        names = []
        for i in range(133):
            temp = []
            names.append(f"cell type {i}")
            cur_ct_df = db[db.cell_type == i].drop(columns=['cell_type'])
            for col in cur_ct_df:
                if col == 'colour':
                    colors.append(cur_ct_df[col].iloc[0])
                else:
                    temp.append(cur_ct_df[col].mean())
            ct_df.append(temp)
        data = pd.DataFrame(ct_df)
        pca = PCA(n_components=2)
    elif content == "Cells (performance hazard)":
        fig.update_layout(title=f't-SNE on each cell')
        colors = db['colour']
        names = list(range(1, 23823))
        data = db.drop(columns=['cell_type', 'colour'])

    tsne = TSNE(n_components=2, random_state=1)
    results_tsne = tsne.fit_transform(data)

    X_tsne = []
    Y_tsne = []
    for pt in results_tsne:
        X_tsne.append(pt[0])
        Y_tsne.append(pt[1])

    if clust_type == "None":
        colors = colors
    elif clust_type == "K-Means":
        colors = kmeans_clustering(X_tsne, Y_tsne, K)
    elif clust_type == "Hierarchical Clustering":
        colors = hierarchical_clustering(X_tsne, Y_tsne, K)
    elif clust_type == "Density-based Clustering":
        colors = densitybased_clustering(X_tsne, Y_tsne, eps, ms)

    fig.add_trace(go.Scatter(x=X_tsne, y=Y_tsne, mode='markers',marker=dict(color=colors), text=names))
    return fig
   

def kmeans_clustering(X, Y, K):
    N = len(X)
    xy = []
    for i in range(N):
        xy.append((X[i], Y[i]))
    labels = np.random.choice(np.arange(K), size = (N,), replace = True)
    rand = rd.sample(range(0,N), K)
    cur_centr = []
    for r in rand:
        cur_centr.append((xy[r]))
    has_changed = True
    count = 0
    while has_changed and count != 100:
        count += 1
        has_changed = False
        for n in range(N):
            min = None
            l = None
            for k in range(K):
                d = np.linalg.norm(np.array(xy[n]) - np.array(cur_centr[k]))
                if min == None:
                    min = d
                    l = k
                else:
                    if min > d:
                        min = d
                        l = k
            if labels[n] != l:
                has_changed = True
                labels[n] = l
        for k in range(K):
            xs = 0
            ys = 0
            div = 0
            for i, l in enumerate(labels):
                if l == k:
                    div += 1
                    xs += xy[i][0]
                    ys += xy[i][1]
            cur_centr[k] = [xs/div, ys/div]
    return labels
def hierarchical_clustering(X, Y, K):
    data = list(zip(X,Y))
    hier_clust = AgglomerativeClustering(n_clusters=K, affinity='manhattan', linkage='average')
    labels = hier_clust.fit_predict(data)
    return labels
def densitybased_clustering(X, Y, eps, ms):
    
    data = list(zip(X,Y))

    """from sklearn.neighbors import NearestNeighbors # importing the library
    neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
    nbrs=neighb.fit(data) # fitting the data to the object
    distances,indices=nbrs.kneighbors(data)
    distances = np.sort(distances, axis = 0) # sorting the distances
    distances = distances[:, 1] # taking the second column of the sorted distances
    plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
    plt.plot(distances) # plotting the distances
    plt.show() # showing the plot"""

    db_clust = DBSCAN(eps=eps, min_samples=ms)
    labels = db_clust.fit_predict(data)
    return labels

@app.callback(
    Output('heatmap_cell_types', 'figure'),
    Input('rfg_heatmap', 'value')
)
def heatmap_cell_types(rfg):

    if rfg:
        db = datarfg
    else:
        db = datainit

    db = db.drop(columns=['colour'])
    ct_df = []
    names = []
    for i in range(133):
        temp = []
        names.append(f"Cell type {i}")
        cur_ct_df = db[db.cell_type == i].drop(columns=['cell_type'])
        for col in cur_ct_df:
            temp.append(cur_ct_df[col].mean())
        ct_df.append(temp)
    data = pd.DataFrame(ct_df)
    data = data.T
    corrs = np.abs(data.corr())
    fig = px.imshow(corrs, template="plotly_dark")
    fig.update_layout(title=f'Correlation heatmap of cell types')
    return fig



if __name__ == "__main__":
    app.run_server(debug=False)