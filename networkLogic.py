import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def getAverageTradeBalance(data,country_list ):

    exports_annual = data.groupby(['Year', 'Export_country'])[['Value (SUM)']].sum().reset_index()
    imports_annual = data.groupby(['Year', 'Import_country'])[['Value (SUM)']].sum().reset_index()

    exports_annual.rename(columns={'Export_country': 'Country', 'Value (SUM)': 'Export_Value'}, inplace=True)
    imports_annual.rename(columns={'Import_country': 'Country', 'Value (SUM)': 'Import_Value'}, inplace=True)

    annual_trade = pd.merge(exports_annual, imports_annual, on=['Year', 'Country'], how='outer')

    annual_trade.fillna(0, inplace=True)
    annual_trade['Trade_Balance'] = annual_trade['Export_Value'] + annual_trade['Import_Value']
    annual_trade.drop(columns=['Export_Value', 'Import_Value'], inplace=True)

    annual_trade = annual_trade.pivot_table(index='Country', columns='Year', values='Trade_Balance')
    annual_trade = annual_trade.loc[country_list]


    avg_balance = annual_trade.mean(axis=1)
    avg_balance_df = pd.DataFrame(avg_balance)
    avg_balance_df.columns = ['Trade_Balance']

    def classify_trade_balance(balance):
        if balance >= 0:
            return 'Exporter'
        elif balance < 0:
            return 'Importer'

    avg_balance_df['Classification'] = avg_balance_df['Trade_Balance'].apply(classify_trade_balance)

    avg_balance_df.reset_index(inplace=True)
    return avg_balance_df



def compute_metrics(graph):
    # Weighted degree centrality
    degree_centrality_weighted = dict(graph.degree(weight='weight'))
    # Unweighted degree centrality
    degree_centrality_unweighted = dict(graph.degree())

    # Calculate average weight per connection by node
    avg_weight_centrality = {node: (degree_centrality_weighted[node] / degree_centrality_unweighted[node]
                                    if degree_centrality_unweighted[node] != 0 else 0)
                             for node in graph.nodes()}

    # Weighted eigenvector centrality
    try:
        eigenvector_centrality_weighted = nx.eigenvector_centrality_numpy(graph, weight='weight')
    except nx.NetworkXError as e:
        print(f"Failed to compute eigenvector centrality: {e}")
        eigenvector_centrality_weighted = {}

    # Unweighted betweenness centrality
    betweenness_centrality_unweighted = nx.betweenness_centrality(graph)
    
    # Unweighted closeness centrality
    closeness_centrality_unweighted = nx.closeness_centrality(graph)

    metrics_df = pd.DataFrame({
        'Weighted_Degree_Centrality': pd.Series(degree_centrality_weighted),
        'Unweighted_Degree_Centrality': pd.Series(degree_centrality_unweighted),
        'Avg_Weight_Centrality': pd.Series(avg_weight_centrality),
        'Weighted_Eigenvector_Centrality': pd.Series(eigenvector_centrality_weighted),
        'Betweenness_Centrality_Unweighted': pd.Series(betweenness_centrality_unweighted),
        'Closeness_Centrality_Unweighted': pd.Series(closeness_centrality_unweighted)
    })

    return metrics_df


def normalize_attribute(G, attribute, scale=1000000000):
    values = np.array([G.nodes[node][attribute] for node in G.nodes()])
    return scale * (values - min(values)) / (max(values) - min(values))

def visualize_full_network(G, trade_balance_df, pos, title, file_name):
    plt.figure(figsize=(15, 10))
    
    ec = nx.eigenvector_centrality_numpy(G, weight='weight')
    nx.set_node_attributes(G, ec, 'centrality')

    normalized_ec = normalize_attribute(G, 'centrality', scale=1)
    
    for _, row in trade_balance_df.iterrows():
        if row['Country'] in G.nodes:
            G.nodes[row['Country']]['trade_balance'] = row['Trade_Balance']
    

    scaling_factor = 60000000
    normalized_te = [G.nodes[node].get('trade_balance', 0) / scaling_factor for node in G.nodes()]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=normalized_te, node_color=normalized_ec, alpha=0.7)
    
    # Add labels to significant nodes based on a size threshold
    significant_nodes = [node for node, size in zip(G.nodes(), normalized_te) if size >  0.75 * np.mean(normalized_te)]
    labels = {node: node for node in significant_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    

    weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(weights.values())
    min_weight = min(weights.values())
    edge_alphas = {e: 0.9 * ((w - min_weight) / (max_weight - min_weight)) for e, w in weights.items()}

    for (u, v), alpha in edge_alphas.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=0.7, alpha=alpha, edge_color='gray')

    cbar = plt.colorbar(nodes)
    cbar.set_label('Eigenvector Centrality')
    
    plt.title(title, fontsize=20)
    plt.annotate('Node color is eigenvector centrality; Node size is value of global exports; Edge transparency is Dependency Strength',
                 xy=(0.5, -0.05), xycoords='axes fraction', ha='center', fontsize=7)
    
    plt.axis('off') 
    plt.savefig(file_name, dpi=300)  
    plt.show()


def visualize_community_network(G, trade_balance_df, pos, title, file_name, spread_factor=1.0):
    fig, ax = plt.subplots(figsize=(14, 9)) 
    
    ec = nx.eigenvector_centrality_numpy(G, weight='weight')
    nx.set_node_attributes(G, ec, 'centrality')

    normalized_ec = normalize_attribute(G, 'centrality', scale=1)
    
    for _, row in trade_balance_df.iterrows():
        if row['Country'] in G.nodes():
            G.nodes[row['Country']]['trade_balance'] = row['Trade_Balance']
    
    scaling_factor = 100000000
    normalized_te = [G.nodes[node].get('trade_balance', 0) / scaling_factor for node in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=normalized_te, node_color='grey', alpha=0.7, ax=ax)

    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, ax=ax)
    
    weights = nx.get_edge_attributes(G, 'weight')
    max_weight = max(weights.values())
    min_weight = min(weights.values())
    norm = plt.Normalize(vmin=min_weight, vmax=max_weight)
    cmap = plt.cm.Blues
    edge_colors = [cmap(norm(weights[(u, v)])) for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=0.7, edge_color=edge_colors, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Mutual Information Strength')

    plt.title(title, fontsize=20)
    plt.annotate('Node size is value of global exports; Edge color represents mutual information strength',
                 xy=(0.5, -0.05), xycoords='axes fraction', ha='center', fontsize=9)
    
    plt.axis('off')
    
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none', transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    
    plt.savefig(file_name, dpi=300, bbox_inches='tight') 
    plt.show()