# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:33:14 2022

@author: DELL
"""
import matplotlib.pyplot as plt
import networkx as nx
G_fb = nx.read_edgelist("Simple_lnd.txt", 
create_using = nx.Graph(), nodetype=str)

print(nx.info(G_fb))

pos = nx.spring_layout(G_fb)
plt.figure(figsize=(20 , 20))
nx.draw_networkx(G_fb, with_labels=False )
plt.axis("off")


# Degree Centrality
pos = nx.spring_layout(G_fb)
betcent= nx.degree_centrality(G_fb)
node_color= [20000.0 * G_fb.degree(v) for v in G_fb]
node_size = [v*10000 for v in betcent.values()]
plt.figure(figsize=(20 , 20))

nx.draw_networkx(G_fb, pos= pos, with_labels=False , node_color= node_color , node_size=node_size)
plt.axis("off")
sorted(betcent, key=betcent.get, reverse=True)[:5]

# betweeness measure

pos = nx.spring_layout(G_fb)
betCent = nx.betweenness_centrality(G_fb, normalized=True, endpoints=True)
node_color = [20000.0 * G_fb.degree(v) for v in G_fb]
node_size = [v * 10000 for v in betCent.values()]

plt.figure(figsize=(20,20))
nx.draw_networkx(G_fb, pos=pos, with_labels =False,
                 node_color=node_color,
                 node_size=node_size)
plt.axis('off')



sorted(betCent, key=betCent.get, reverse=True)[:5]

pos = nx.spring_layout(G_fb)
betCent = nx.degree_centrality(G_fb)
node_color = [20000.0 * G_fb.degree(v) for v in G_fb]
node_size = [v * 10000 for v in betCent.values()]

plt.figure(figsize=(20,20))
nx.draw_networkx(G_fb, pos=pos, with_labels =False,
                 node_color=node_color,
                 node_size=node_size)
plt.axis('off')




eigcent= nx.eigenvector_centrality(G_fb)
#print(eigcent)

node_color= [20000.0 * G_fb.degree(v) for v in G_fb]
node_size = [v*10000 for v in eigcent.values()]

plt.figure(figsize=(20 , 20))

nx.draw_networkx(G_fb, pos= pos, with_labels=False , node_color= node_color , node_size=node_size)

plt.axis("off")
sorted(eigcent, key=eigcent.get, reverse=True)[:5]

# Eccentricity 

#nx.eccentricity(WeightedG, 'Wolverhampton')

nx.degree_centrality(G_fb)

nx.eccentricity(G_fb)

nx.eigenvector_centrality(G_fb)

nx.betweenness_centrality(G_fb) 

#Number of Community

import networkx as nx
from cdlib import algorithms, viz


G_fb = nx.read_edgelist("Simple_lnd.txt" , create_using = nx.Graph(), nodetype=int)

#detecting communities
coms = algorithms.louvain(G_fb, weight = "weight", resolution = 1.)

pos = nx.spring_layout(G_fb)

viz.plot_network_clusters(G_fb, coms, pos)
viz.plot_community_graph(G_fb, coms)

len(coms.communities)

coms.communities