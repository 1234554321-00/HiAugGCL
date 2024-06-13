
import os
import pandas as pd
import networkx as nx
import math
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import torch_geometric.utils as utils
from dgl.nn.pytorch import GATConv
from torch_geometric.utils import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
import uuid
import matplotlib.pyplot as plt
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.optim as optim
import dgl.function as fn
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score, ndcg_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import zipfile
from sklearn.metrics import ndcg_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF
import re
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import fbeta_score, average_precision_score
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from torch.utils.data import DataLoader, TensorDataset


def process_data(folder_path):
    ground_truth_ratings = None
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
        'movie_directors.xlsx': ['movieID', 'directorID'],
        'movie_actors.xlsx': ['movieID', 'actorID'],
        'movie_genres.xlsx': ['movieID', 'genreID']
    }

    unique_values = {column: set() for column in file_columns.keys()}

    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path, usecols=columns)
            for column in columns:
                if column in unique_values and column != 'rating':
                    unique_values[column].update(df[column].unique())
            if 'rating' in columns:
                if ground_truth_ratings is None:
                    ground_truth_ratings = df
                else:
                    ground_truth_ratings = pd.concat([ground_truth_ratings, df], ignore_index=True)
        else:
            print(f"File not found: {file_name}")

    return ground_truth_ratings

def create_heterogeneous_graph(folder_path):
    # Create an empty graph
    G = nx.Graph()
    # Create dictionaries to store the number of nodes for each node type
    node_counts = {'userID': 0, 'movieID': 0, 'directorID': 0, 'actorID': 0, 'genreID': 0}

    # Create a dictionary to store mapping between nodes and their attributes
    node_attributes = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create dictionaries to store the number of nodes and edges for each type of relationship
    relationship_counts = {}

    # Create a dictionary to map each file to its corresponding columns
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
        'movie_directors.xlsx': ['movieID', 'directorID'],
        'movie_actors.xlsx': ['movieID', 'actorID'],
        'movie_genres.xlsx': ['movieID', 'genreID']
    }

    # Iterate through the files and read them to populate the graph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Add nodes and edges to the graph based on the file's content
            if 'userID' in columns:
                for _, row in df.iterrows():
                    user_node = f"userID:{row['userID']}"
                    movie_node = f"movieID:{row['movieID']}"
                    rating = row['rating']

                    # Add nodes only if they don't exist
                    if user_node not in G:
                        G.add_node(user_node, type='userID')
                        node_counts['userID'] += 1

                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    G.add_edge(user_node, movie_node, weight=rating)

            if 'directorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    director_node = f"directorID:{row['directorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if director_node not in G:
                        G.add_node(director_node, type='directorID')
                        node_counts['directorID'] += 1

                    G.add_edge(movie_node, director_node)

            if 'actorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    actor_node = f"actorID:{row['actorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if actor_node not in G:
                        G.add_node(actor_node, type='actorID')
                        node_counts['actorID'] += 1

                    G.add_edge(movie_node, actor_node)

            if 'genreID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    genre_node = f"genreID:{row['genreID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if genre_node not in G:
                        G.add_node(genre_node, type='genreID')
                        node_counts['genreID'] += 1

                    G.add_edge(movie_node, genre_node)

    # Print the number of nodes and edges for the graph and the node counts
    print("Graph information:")
    print("Nodes:", len(G.nodes()))
    print("Edges:", len(G.edges()))
    for node_type, count in node_counts.items():
        print(f"Number of {node_type} nodes: {count}")

    return G

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-User--------------------------------------------
#****************************************************************************************
def hypergraph_MU(folder_path):

    # Create an empty hypergraph
    hyper_MU = {}
    relationship_counts = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MU = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create a dictionary to map the 'user_movies.xlsx' file to its corresponding columns
    file_columns = {
        'user_movies.xlsx': ['userID', 'movieID', 'rating'],
    }

    # Iterate through the files and read them to populate the hypergraph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hypergraph and relationship counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                user_node = f"userID:{str(row['userID'])}"
                rating = row['rating']

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MU:
                    hyper_MU[movie_node] = []

                # Add the user node to the hypergraph if it doesn't exist
                if user_node not in hyper_MU:
                    hyper_MU[user_node] = []

                # Add the user node to the movie hyperedge
                hyper_MU[movie_node].append(user_node)

                # Set the type attribute in att_MU
                att_MU[user_node] = {'type': 'userID'}
                att_MU[movie_node] = {'type': 'movieID'}

                edge_weights[(movie_node, user_node)] = rating

                # Count nodes and edges for the userID-movieID relationship
                relationship = 'userID-movieID'
                relationship_counts[relationship] = relationship_counts.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts[relationship]['nodes'] += 2  # Two nodes (movie and user)
                relationship_counts[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MU = {k: v for k, v in hyper_MU.items() if v}
    
    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MU.values())

    print("Hypergraph information of MU:")
    print("Number of hyperedges of MU (nodes):", len(hyper_MU))
    print("Number of edges of MU:", num_edges)

    return hyper_MU, att_MU


#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-Director--------------------------------------------
#****************************************************************************************
def hypergraph_MD(folder_path):
 
    # Create an empty hyper_MD
    hyper_MD = {}
    relationship_counts_MD = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MD = {}
    
    # Create a dictionary to map the 'director_movies.xlsx' file to its corresponding columns
    file_columns = {
        'movie_directors.xlsx': ['movieID', 'directorID'],
    }

    # Iterate through the files and read them to populate the hyper_MD
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MD and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                director_node = f"directorID:{str(row['directorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MD:
                    hyper_MD[movie_node] = []

                # Add the director node to the hyper_MD if it doesn't exist
                if director_node not in hyper_MD:
                    hyper_MD[director_node] = []

                # Add the director node to the movie hyperedge
                hyper_MD[movie_node].append(director_node)

                # Set the type attribute in att_MD
                att_MD[director_node] = {'type': 'directorID'}
                att_MD[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the directorID-movieID relationship
                relationship = 'directorID-movieID'
                relationship_counts_MD[relationship] = relationship_counts_MD.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MD[relationship]['nodes'] += 2  # Two nodes (movie and director)
                relationship_counts_MD[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MD = {k: v for k, v in hyper_MD.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MD.values())

    print("Hypergraph information of MD:")
    print("Number of hyperedges of MD (nodes):", len(hyper_MD))
    print("Number of edges of MD:", num_edges)

    return hyper_MD, att_MD

def generate_incidence_matrices_MD(hyper_MD, att_MD):
    """
    Generates incidence matrices for movies and directors.

    Args:
        hyper_MD (dict): Hypergraph representing connections between movies and directors.
        att_MD (dict): Dictionary containing attributes for nodes.

    Returns:
        tuple: A tuple containing the movie-director incidence matrix and its transpose.
    """
    movie_nodes = [node for node in att_MD if att_MD[node]['type'] == 'movieID']
    director_nodes = [node for node in att_MD if att_MD[node]['type'] == 'directorID']

    num_movies = len(movie_nodes)
    num_directors = len(director_nodes)
    incidence_matrix_MD = np.zeros((num_directors, num_movies), dtype=float)  # Swap dimensions

    for movie_index, movie_node in enumerate(movie_nodes):
        directors_connected = hyper_MD.get(movie_node, [])
        for director_node in directors_connected:
            if director_node in director_nodes:
                director_index = director_nodes.index(director_node)
                incidence_matrix_MD[director_index, movie_index] = 1  # Swap indices
    
    print("incidence_matrix_MD Shape", incidence_matrix_MD.shape)
    
    return incidence_matrix_MD

#****************************************************************************************
#----------------------------------- Hypergraph and Incidence Matrices Movie-Actor--------------------------------------------
#****************************************************************************************
def hypergraph_MA(folder_path):
    """
    Generate a hypergraph based on the files found in the specified folder path.

    Args:
    - folder_path (str): Path to the folder containing the files.

    Returns: 
    - hyper_MA (dict): Dictionary representing the hypergraph.
    - att_MA (dict): Dictionary containing attributes of nodes in the hypergraph.
    """
    # Create an empty hyper_MA
    hyper_MA = {}
    relationship_counts_MA = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MA = {}
    
    # Create a dictionary to map the 'actor_movies.xlsx' file to its corresponding columns
    file_columns = {
        'movie_actors.xlsx': ['movieID', 'actorID'],
    }

    # Iterate through the files and read them to populate the hyper_MA
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MA and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                actor_node = f"actorID:{str(row['actorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MA:
                    hyper_MA[movie_node] = []

                # Add the actor node to the hyper_MA if it doesn't exist
                if actor_node not in hyper_MA:
                    hyper_MA[actor_node] = []

                # Add the actor node to the movie hyperedge
                hyper_MA[movie_node].append(actor_node)

                # Set the type attribute in att_MA
                att_MA[actor_node] = {'type': 'actorID'}
                att_MA[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the actorID-movieID relationship
                relationship = 'actorID-movieID'
                relationship_counts_MA[relationship] = relationship_counts_MA.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MA[relationship]['nodes'] += 2  # Two nodes (movie and actor)
                relationship_counts_MA[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MA = {k: v for k, v in hyper_MA.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MA.values())

    print("Hypergraph information of MA:")
    print("Number of hyperedges of MA (nodes):", len(hyper_MA))
    print("Number of edges of MA:", num_edges)

    return hyper_MA, att_MA

def generate_incidence_matrices_MA(hyper_MA, att_MA):

    # Extract movie and actor nodes
    movie_nodes = [node for node in att_MA if att_MA[node]['type'] == 'movieID']
    actor_nodes = [node for node in att_MA if att_MA[node]['type'] == 'actorID']

    # Initialize incidence matrix
    num_movies = len(movie_nodes)
    num_actors = len(actor_nodes)
    incidence_matrix_MA = np.zeros((num_actors, num_movies), dtype=float)

    # Populate the incidence matrix
    for movie_index, movie_node in enumerate(movie_nodes):
        actors_connected = hyper_MA.get(movie_node, [])
        for actor_node in actors_connected:
            if actor_node in actor_nodes:
                actor_index = actor_nodes.index(actor_node)
                incidence_matrix_MA[actor_index, movie_index] = 1  # Adjust based on your hypergraph structure

    print("incidence_matrix_MA Shape", incidence_matrix_MA.shape)
    
    return incidence_matrix_MA

#****************************************************************************************
#----------------------------------- Hypergraph Convolutional Embedding --------------------------------------------
#****************************************************************************************
def generate_incidence_matrices_MU(hyper_MU, att_MU):
    """
    Generates incidence matrices for movies and users.

    Args:
        hyper_MU (dict): Hypergraph representing connections between movies and users.
        att_MU (dict): Dictionary containing attributes for nodes.

    Returns:
        tuple: A tuple containing the movie-user incidence matrix and its transpose.
    """
    movie_nodes = [node for node in att_MU if att_MU[node]['type'] == 'movieID']
    user_nodes = [node for node in att_MU if att_MU[node]['type'] == 'userID']

    num_movies = len(movie_nodes)
    num_users = len(user_nodes)
    incidence_matrix_MU = np.zeros((num_users, num_movies), dtype=float)

    for movie_index, movie_node in enumerate(movie_nodes):
        users_connected = hyper_MU.get(movie_node, [])
        for user_node in users_connected:
            if user_node in user_nodes:
                user_index = user_nodes.index(user_node)
                incidence_matrix_MU[user_index, movie_index] = 1

    print("incidence_matrix_MU Shape", incidence_matrix_MU.shape)
    
    return incidence_matrix_MU

def enrich_incidence_matrices_MUD(hyper_MU, att_MU, hyper_MD):
    """
    Enriches the incidence matrix for movies and users based on directors.

    Args:
        hyper_MU (dict): Hypergraph representing connections between movies and users.
        att_MU (dict): Dictionary containing attributes for nodes.
        hyper_MD (dict): Hypergraph representing connections between movies and directors.

    Returns:
        numpy.ndarray: Enriched incidence matrix for movies and users.
    """
    movie_nodes = [node for node in att_MU if att_MU[node]['type'] == 'movieID']
    user_nodes = [node for node in att_MU if att_MU[node]['type'] == 'userID']

    num_movies = len(movie_nodes)
    num_users = len(user_nodes)
    incidence_matrix_MUD = np.zeros((num_users, num_movies), dtype=float)

    # Create a dictionary to store movies associated with each director
    director_movies = {}
    for movie, directors in hyper_MD.items():
        for director in directors:
            director_movies.setdefault(director, []).append(movie)

    for movie_index, movie_node in enumerate(movie_nodes):
        users_connected = hyper_MU.get(movie_node, [])
        for user_node in users_connected:
            if user_node in user_nodes:
                user_index = user_nodes.index(user_node)
                incidence_matrix_MUD[user_index, movie_index] = 1 

        # Enrich matrix by connecting users to movies associated with the director
        for director_node in hyper_MD.get(movie_node, []):
            for movie_director_connected in director_movies.get(director_node, []):
                movie_index_director = movie_nodes.index(movie_director_connected)
                incidence_matrix_MUD[user_index, movie_index_director] = 1

    return incidence_matrix_MUD

def enrich_incidence_matrices_MUA(hyper_MU, att_MU, hyper_MA):
    """
    Enriches the incidence matrix for movies and users based on actors.

    Args:
        hyper_MU (dict): Hypergraph representing connections between movies and users.
        att_MU (dict): Dictionary containing attributes for nodes.
        hyper_MA (dict): Hypergraph representing connections between movies and actors.

    Returns:
        numpy.ndarray: Enriched incidence matrix for movies and users.
    """
    # Extract movie and user nodes
    movie_nodes = [node for node in att_MU if att_MU[node]['type'] == 'movieID']
    user_nodes = [node for node in att_MU if att_MU[node]['type'] == 'userID']

    # Initialize incidence matrix
    num_movies = len(movie_nodes)
    num_users = len(user_nodes)
    incidence_matrix_MUA = np.zeros((num_users, num_movies), dtype=float)

    # Create a dictionary to store movies associated with each actor
    actor_movies = {}
    for movie, actors in hyper_MA.items():
        for actor in actors:
            actor_movies.setdefault(actor, []).append(movie)

    # Populate the incidence matrix
    for movie_index, movie_node in enumerate(movie_nodes):
        users_connected = hyper_MU.get(movie_node, [])
        for user_node in users_connected:
            if user_node in user_nodes:
                user_index = user_nodes.index(user_node)
                incidence_matrix_MUA[user_index, movie_index] = 1 

        # Enrich the matrix with users who share a movie based on the actor
        for actor_node in hyper_MA.get(movie_node, []):
            for movie_actor_connected in actor_movies.get(actor_node, []):
                movie_index_actor = movie_nodes.index(movie_actor_connected)
                incidence_matrix_MUA[user_index, movie_index_actor] = 1  # Connect users to movies associated with the actor

    return incidence_matrix_MUA


def attention_cosine_similarity(incidence_matrix):
    # Check if the incidence matrix is empty
    if incidence_matrix.shape[0] == 0:
        raise ValueError("Input incidence matrix has zero samples.")
    
    # Calculate cosine similarity between rows of the incidence matrix
    cosine_sim = cosine_similarity(incidence_matrix)
    
    # Apply LeakyReLU activation
    activated_cosine_sim = leaky_relu(cosine_sim)
    
    # Apply softmax function to obtain attention weights
    attention_weights = np.exp(activated_cosine_sim) / np.sum(np.exp(activated_cosine_sim), axis=1, keepdims=True)
    
    return attention_weights

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def compute_hypergraph_laplacian_MU(att_MU, hyper_MU):
    # Generate incidence matrices
    incidence_matrix_MU = generate_incidence_matrices_MU(hyper_MU, att_MU)

    # Compute attention weights using attention-based cosine similarity
    attention_weights = attention_cosine_similarity(incidence_matrix_MU)

    # Compute Laplacian matrix
    Laplacian_MU = dynamic_laplacian(incidence_matrix_MU, attention_weights)

    return Laplacian_MU

def compute_hypergraph_laplacian_MUD(hyper_MU, att_MU, hyper_MD):
    # Generate incidence matrices
    incidence_matrix_MUD = enrich_incidence_matrices_MUD(hyper_MU, att_MU, hyper_MD)

    # Compute attention weights using attention-based cosine similarity
    attention_weights = attention_cosine_similarity(incidence_matrix_MUD)

    # Compute Laplacian matrix
    Laplacian_MUD = dynamic_laplacian(incidence_matrix_MUD, attention_weights)

    return Laplacian_MUD

def compute_hypergraph_laplacian_MUA(hyper_MU, att_MU, hyper_MA):
    # Generate incidence matrices
    incidence_matrix_MUA = enrich_incidence_matrices_MUA(hyper_MU, att_MU, hyper_MA)

    # Compute attention weights using attention-based cosine similarity
    attention_weights = attention_cosine_similarity(incidence_matrix_MUA)

    # Compute Laplacian matrix
    Laplacian_MUA = dynamic_laplacian(incidence_matrix_MUA, attention_weights)

    return Laplacian_MUA

def dynamic_laplacian(incidence_matrix, attention_weights):
    # Compute Laplacian matrix based on attention weights
    weight_matrix = np.diag(np.sum(attention_weights, axis=1)) - attention_weights
    dynamic_Laplacian = incidence_matrix.T @ weight_matrix @ incidence_matrix
    
    # Normalize Laplacian matrix to have the same dimensions
    dynamic_Laplacian = dynamic_Laplacian / np.max(dynamic_Laplacian)
    
    # Print the dynamic Laplacian matrix
    print("Dynamic Laplacian Matrix:")
    print(dynamic_Laplacian)
    
    return dynamic_Laplacian

def fuse_matrix1_by_weight(Laplacian_MU, Laplacian_MUD):
    # Compute cosine similarity between the two matrices
    similarity_matrix = cosine_similarity(Laplacian_MU.flatten().reshape(1, -1), Laplacian_MUD.flatten().reshape(1, -1))
    alpha = similarity_matrix[0, 0]
    
    # Fuse the matrices based on the calculated alpha
    fused_auggmention1 = alpha * Laplacian_MU + (1 - alpha) * Laplacian_MUD
    
    return fused_auggmention1

def fuse_matrix2_by_weight(fused_auggmention1, Laplacian_MUA):
    # Compute cosine similarity between the two matrices
    similarity_matrix = cosine_similarity(fused_auggmention1.flatten().reshape(1, -1), Laplacian_MUA.flatten().reshape(1, -1))
    alpha = similarity_matrix[0, 0]
    
    # Fuse the matrices based on the calculated alpha
    fused_auggmention2 = alpha * fused_auggmention1 + (1 - alpha) * Laplacian_MUA
    
    return fused_auggmention2


#----------------------------------------------------------------------------------------------------------------
#--------------------------------------------- ***LightGCL Model*** -------------------------------------------
#----------------------------------------------------------------------------------------------------------------

class GCLLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCLLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, Laplacian, features):
        support = self.linear(features)
        output = torch.sparse.mm(Laplacian, support)
        return output

class HierarchicalModel(nn.Module):
    def __init__(self, num_relations, embedding_dim):
        super(HierarchicalModel, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gcl_layers = nn.ModuleList([GCLLayer(embedding_dim * 2, embedding_dim) for _ in range(2)])  # 2 message passing layers

    def forward(self, Laplacian_list):
        num_nodes = Laplacian_list[0].shape[0]  # Assuming Laplacian is square
        embeddings_list = []

        for i, Laplacian in enumerate(Laplacian_list):
            embedding = torch.eye(num_nodes, device=Laplacian.device)  # Initialize embeddings as identity matrix
            for layer in self.gcl_layers:
                embedding = layer(Laplacian, embedding)
            embeddings_list.append(embedding)

        # Split embeddings for users and items
        half_num_nodes = num_nodes // 2
        embedding_user, embedding_item = torch.split(embeddings_list[-1], [half_num_nodes, num_nodes - half_num_nodes], dim=0)

        return embedding_user, embedding_item, embeddings_list

def InfoNCE_loss(embeddings, positive_pairs, negative_pairs, temperature=0.5):
    # Compute similarities
    sim_positive = F.cosine_similarity(embeddings.unsqueeze(1), positive_pairs.unsqueeze(0), dim=-1)
    sim_negative = F.cosine_similarity(embeddings.unsqueeze(1), negative_pairs.unsqueeze(0), dim=-1)
    
    # Compute numerator and denominator for InfoNCE loss
    numerator = torch.exp(sim_positive / temperature)
    denominator = numerator + torch.sum(torch.exp(sim_negative / temperature), dim=-1)
    
    # Compute InfoNCE loss
    loss = -torch.log(numerator / denominator).mean()
    
    return loss

def train_model(Laplacian_matrices_list, num_epochs=500, learning_rate=0.01, temperature=0.5):
    num_relations = len(Laplacian_matrices_list)
    embedding_dim = 128  
    num_samples = Laplacian_matrices_list[0][0].shape[0]

    # Instantiate the HierarchicalModel
    model = HierarchicalModel(num_relations, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Forward pass for Laplacian Matrices
        embedding_user, embedding_item, _ = model(Laplacian_matrices_list)

        # Prepare positive and negative pairs for InfoNCE loss
        positive_user_pairs = embedding_user[:num_samples].unsqueeze(1)
        positive_item_pairs = embedding_item[num_samples:].unsqueeze(1)
        negative_user_pairs = embedding_user[:num_samples].unsqueeze(0)
        negative_item_pairs = embedding_item[num_samples:].unsqueeze(0)

        # Compute InfoNCE loss for users and items separately
        user_loss = InfoNCE_loss(embedding_user[:num_samples], positive_user_pairs, negative_user_pairs, temperature)
        item_loss = InfoNCE_loss(embedding_item[num_samples:], positive_item_pairs, negative_item_pairs, temperature)

        # Total loss is the sum of user and item losses
        total_loss = user_loss + item_loss

        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print training progress
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item()}")

    return model

def generate_fusion_embeddings(model, Laplacian_matrices_list):
    # Generate fusion embeddings using the trained model
    fusion_embeddings_list = []
    for Laplacian_list in Laplacian_matrices_list:
        attention_embedding_user, attention_embedding_item, _, _ = model(Laplacian_list)
        fusion_embeddings_list.append(torch.cat((attention_embedding_user, attention_embedding_item), dim=0))
        
    print("fusion Embeddings List:", fusion_embeddings_list)
    
    return fusion_embeddings_list

def create_ground_truth_rating_matrix(folder_path):
    # Use the process_data function to obtain ground truth ratings DataFrame
    ground_truth_ratings = process_data(folder_path)

    # Print unique value counts
    unique_values = {
        'userID': ground_truth_ratings['userID'].nunique(),
        'movieID': ground_truth_ratings['movieID'].nunique(),
    }
    print("Unique value counts:")
    for column, value in unique_values.items():
        print(f"{column}: {value}")

    print("ground Truth Ratings:", ground_truth_ratings)

    return ground_truth_ratings

def generate_recommendations(fusion_embeddings_list, topk=10):
    recommendations = {}

    # Calculate cosine similarity between embeddings
    num_users = fusion_embeddings_list[0].shape[0] // 2  # Assuming half nodes are users
    num_items = fusion_embeddings_list[0].shape[0] - num_users
    for user_id in range(num_users):
        user_embedding = fusion_embeddings_list[user_id]
        item_embeddings = fusion_embeddings_list[num_users:]

        # Check if item_embeddings is empty
        if not item_embeddings:
            return {}

        # Calculate similarity between the target user and all users
        user_similarities = cosine_similarity(user_embedding.reshape(1, -1), torch.cat(fusion_embeddings_list[:num_users]))
        
        # Calculate similarity between the target item and all items
        item_similarities = cosine_similarity(user_embedding.reshape(1, -1), torch.cat(fusion_embeddings_list[num_users:]))
        
        # Combine user and item similarities into a single score
        combined_similarities = user_similarities + item_similarities
        
        # Get top-k indices of most similar items
        topk_indices = combined_similarities.argsort()[0][-topk:][::-1]
        
        # Map top-k indices to item IDs
        recommended_items = [num_users + idx.item() for idx in topk_indices]
        recommendations[user_id] = recommended_items

    return recommendations

def split_and_save_data(ground_truth_ratings, test_size=0.2, random_state=42):
    # Split the ground truth ratings into train and test sets
    train_data, test_data = train_test_split(ground_truth_ratings, test_size=test_size, random_state=random_state)
    return train_data, test_data

def evaluate_RS_Model_Prediction(movie_embeddings, folder_path, test_size=0.2, random_state=42):
    # Get ground truth ratings DataFrame
    ground_truth_ratings = create_ground_truth_rating_matrix(folder_path)

    # Split ground truth ratings into train and test sets
    train_data, test_data = split_and_save_data(ground_truth_ratings, test_size=test_size, random_state=random_state)

    # Get the movie indices present in movie_embeddings
    movie_indices = set(range(len(movie_embeddings)))

    # Filter train_data and test_data to include only movie indices present in movie_embeddings
    train_data = train_data[train_data['movieID'].astype(int).isin(movie_indices)]
    test_data = test_data[test_data['movieID'].astype(int).isin(movie_indices)]

    # Extract relevant movie IDs from train_data
    train_movie_ids = train_data['movieID'].astype(int).values

    # Filter movie_embeddings to include only relevant movie IDs
    train_X = movie_embeddings[train_movie_ids]

    # Prepare training labels
    train_y = train_data['rating'].values

    # Instantiate and train the SVR model
    svr_model = SVR()
    svr_model.fit(train_X.detach().numpy(), train_y)  # Adjusted this line to detach the tensor before converting to a NumPy array

    # Prepare test data
    test_movie_ids = test_data['movieID'].values.astype(int)
    test_X = movie_embeddings[test_movie_ids]

    # Make predictions for test data using the SVR model
    test_predictions = svr_model.predict(test_X.detach().numpy())  # Detach before converting to NumPy

    # Calculate MAE and RMSE for test data
    test_mae = mean_absolute_error(test_data['rating'], test_predictions)
    test_rmse = mean_squared_error(test_data['rating'], test_predictions, squared=False)
    print("MAE for test data (SVR):", test_mae)
    print("RMSE for test data (SVR):", test_rmse)

    return test_mae, test_rmse

def main():
    # Define your file paths for different datasets
    folder_path = 'C:\\IMDB'
    
    # Create and analyze the traditional graph
    graph = create_heterogeneous_graph(folder_path)

    # Call the function and store the hypergraphs
    hyper_MU, att_MU = hypergraph_MU(folder_path)  
    hyper_MD, att_MD = hypergraph_MD(folder_path) 
    hyper_MA, att_MA = hypergraph_MA(folder_path)
    
    # Compute dynamic Laplacian matrix for Enriched_incidence_matrices_MUD
    Laplacian_MU = compute_hypergraph_laplacian_MU(hyper_MU, att_MU)
    Laplacian_MUD = compute_hypergraph_laplacian_MUD(hyper_MU, att_MU, hyper_MD)
    Laplacian_MUA = compute_hypergraph_laplacian_MUA(hyper_MU, att_MU, hyper_MA)
    fused_auggmention1 = fuse_matrix1_by_weight(Laplacian_MU, Laplacian_MUD)
    fused_auggmention2 = fuse_matrix2_by_weight(fused_auggmention1, Laplacian_MUA)
        
    # Convert Laplacian matrices to tensors
    Laplacian_MU = [torch.tensor(Laplacian_MU, dtype=torch.float32, requires_grad=True)] 
    Laplacian_MUD = [torch.tensor(Laplacian_MUD, dtype=torch.float32, requires_grad=True)] 
    Laplacian_MUA = [torch.tensor(Laplacian_MUA, dtype=torch.float32, requires_grad=True)] 
    fused_auggmention1 = [torch.tensor(fused_auggmention1, dtype=torch.float32, requires_grad=True)] 
    fused_auggmention2 = [torch.tensor(fused_auggmention2, dtype=torch.float32, requires_grad=True)] 

    # train_model([Laplacian_MUD, Laplacian_MUA, Laplacian_MUG])
    model = train_model([Laplacian_MU,Laplacian_MUD, Laplacian_MUA, fused_auggmention1, fused_auggmention2], num_epochs=500, learning_rate=0.01)
    fusion_embeddings_list = generate_fusion_embeddings(model, [Laplacian_MU,Laplacian_MUD, Laplacian_MUA, fused_auggmention1, fused_auggmention2])
    ground_truth_ratings = create_ground_truth_rating_matrix(folder_path)
    
    recommendations = generate_recommendations(fusion_embeddings_list)
    # print(recommendations)
    train_ratings, test_ratings = train_test_split(ground_truth_ratings, test_size=0.2, random_state=42)
    train_recommendations = generate_recommendations(fusion_embeddings_list)
    # Evaluate recommendations based on the testing set
    test_mae, test_rmse = evaluate_RS_Model_Prediction(fusion_embeddings_list, folder_path, test_size=0.2, random_state=42)

if __name__ == "__main__":
    main()