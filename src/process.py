import logging
import typing as tp
from collections import defaultdict

import hydra
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from tqdm import tqdm


def get_aggregated_sessions_data(events: pd.DataFrame, session_column: str, user_column: str, item_column: str, transaction_type_column: str) -> pd.DataFrame:
    sessions = events.groupby(session_column).agg(
        {user_column: 'first', item_column: list, transaction_type_column: list}
        )
    return sessions

def prepare_dataset_for_sequential_model(sessions: pd.DataFrame, item_column: str, transaction_type_column: str, target_column: str) -> pd.DataFrame:
    sessions[target_column] = sessions[item_column].map(lambda x: x[-1:])
    sessions[item_column] = sessions[item_column].map(lambda x: x[:-1])
    sessions[transaction_type_column] = sessions[transaction_type_column].map(lambda x: x[:-1])
    return sessions[sessions[item_column].map(len)>1]

def calculate_frequencies(sessions: pd.DataFrame, item_column: str) -> tp.Dict[str, int]:
    pair_frequency = defaultdict(int)
    item_frequency = defaultdict(int)

    for group in tqdm(sessions[item_column],
                        position=0,
                        leave=True,
                        desc="Compute item rating frequency"):
    # Get a list of movies rated by the user.
        current_movies = group
        for i in range(len(current_movies)):
            item_frequency[current_movies[i]] += 1
            for j in range(i + 1, len(current_movies)):
                x = min(current_movies[i], current_movies[j])
                y = max(current_movies[i], current_movies[j])
                pair_frequency[(x, y)] += 1

    return item_frequency, pair_frequency

def construct_knowledge_graph(item_frequency: tp.Dict[str, int], pair_frequency: tp.Dict[str, int], weight_threshold: int = 10) -> nx.Graph:
    D = np.log(sum(item_frequency.values()))

    # Create the movies undirected graph.
    item_graph = nx.Graph()
    # Add weighted edges between movies.
    # This automatically adds the movie nodes to the graph.
    for pair in tqdm(
        pair_frequency, position=0, leave=True, desc="Creating the item graph"
    ):
        x, y = pair
        xy_frequency = pair_frequency[pair]
        x_frequency = item_frequency[x]
        y_frequency = item_frequency[y]
        pmi = np.log(xy_frequency) - np.log(x_frequency) - np.log(y_frequency) + D
        weight = pmi * xy_frequency
        if weight >= weight_threshold:
            item_graph.add_edge(x, y, weight=weight)

    return item_graph

def random_walk_next_step(graph: nx.Graph, previous: int, current: int, p: float, q: float) -> int:
    neighbors = list(graph.neighbors(current))

    weights = []
    # Adjust the weights of the edges to the neighbors with respect to p and q.
    for neighbor in neighbors:
        if neighbor == previous:
            # Control the probability to return to the previous node.
            weights.append(graph[current][neighbor]["weight"] / p)
        elif graph.has_edge(neighbor, previous):
            # The probability of visiting a local node.
            weights.append(graph[current][neighbor]["weight"])
        else:
            # Control the probability to move forward.
            weights.append(graph[current][neighbor]["weight"] / q)

    # Compute the probabilities of visiting each neighbor.
    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]
    # Probabilistically select a neighbor to visit.
    next = np.random.choice(neighbors, size=1, p=probabilities)[0]
    return next

def construct_random_walks(graph: nx.Graph, nodes_ordered: tp.List[int], num_walks: int, num_steps: int, p: float, q: float) -> tp.List[int]:
    walks = []
    # Perform multiple iterations of the random walk.
    for walk_iteration in range(num_walks):
        np.random.default_rng(0).shuffle(nodes_ordered)

        for node in tqdm(
            nodes_ordered,
            position=0,
            leave=True,
            desc=f"Random walks iteration {walk_iteration + 1} of {num_walks}",
        ):
            # Start the walk with a random node from the graph.
            walk = [node]
            # Randomly walk for num_steps.
            while len(walk) < num_steps:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # Compute the next node to visit.
                next = random_walk_next_step(graph, previous, current, p, q)
                walk.append(next)
            # Add the walk to the generated sequence.
            walks.append(walk)
    return walks

def generate_dataset_for_node2vec_model(sequences: tp.List[int], window_size: int, num_negative_samples: int, vocabulary_size: int) -> tf.data.Dataset:
    example_weights = defaultdict(int)
    # Iterate over all sequences (walks).
    for sequence in tqdm(
        sequences,
        position=0,
        leave=True,
        desc=f"Generating postive and negative examples",
    ):
        # Generate positive and negative skip-gram pairs for a sequence (walk).
        pairs, labels = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            negative_samples=num_negative_samples,
        )
        for idx in range(len(pairs)):
            pair = pairs[idx]
            label = labels[idx]
            target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
            if target == context:
                continue
            entry = (target, context, label)
            example_weights[entry] += 1

    targets, contexts, labels, weights = [], [], [], []
    for entry in tqdm(
        example_weights,
        position=0,
        leave=True,
        desc=f"Populating the dataset"
    ):
        weight = example_weights[entry]
        target, context, label = entry
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        weights.append(weight)
    
    inputs = {
        "target": np.array(targets),
        "context": np.array(contexts),
    }

    dataset = tf.data.Dataset.from_tensor_slices((inputs, np.array(labels), np.array(weights)))
    
    return dataset

@hydra.main(config_path="../config", config_name='main')
def process_data(config: DictConfig):
    """Function to process the data"""
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
            filename=config.logging.path + 'process_data.txt',
            filemode='w'
        )
    
    raw_path = abspath(config.raw.path)
    savefile_path = abspath(config.processed.path) + "/"
    logging.info(f"Process data using {raw_path}")

    timestamp_column = config.process.timestamp_column
    user_column = config.process.user_column
    session_column = config.process.session_column
    transaction_type_column = config.process.transaction_type_column
    item_column = config.process.item_column
    target_column = config.process.target_column
    train_ratio = config.process.train_test_ratio

    events = pd.read_csv(raw_path)
    logging.info(f"Constructing KG, train and test datasets {raw_path}")
    events[timestamp_column] = pd.to_datetime(events[timestamp_column])
    datetime_for_knowledge_graph = events[timestamp_column].quantile(train_ratio/2).date()
    datetime_for_train_test = events[timestamp_column].quantile(train_ratio).date()
    events_for_kg = events[events[timestamp_column].dt.date<=datetime_for_knowledge_graph]
    events_for_train = events[events[timestamp_column].dt.date<=datetime_for_train_test]
    events_for_test = events[events[timestamp_column].dt.date>datetime_for_train_test]
    
    user_vocabulary = events_for_train[user_column].unique()
    np.save(savefile_path + user_column, user_vocabulary)
    logging.info(f"User vocabulary saved to {savefile_path + user_column}")

    transaction_type_vocabulary = events_for_train[transaction_type_column].unique()
    np.save(savefile_path + transaction_type_column, transaction_type_vocabulary)
    logging.info(f"Transaction vocabulary saved to {savefile_path + transaction_type_column}")


    sessions_for_kg = get_aggregated_sessions_data(events_for_kg, session_column, user_column, item_column, transaction_type_column)
    sessions_for_train = get_aggregated_sessions_data(events_for_train, session_column, user_column, item_column, transaction_type_column)
    sessions_for_train = prepare_dataset_for_sequential_model(sessions_for_train, item_column, transaction_type_column, target_column)
    sessions_for_test = get_aggregated_sessions_data(events_for_train, session_column, user_column, item_column, transaction_type_column)
    sessions_for_test = prepare_dataset_for_sequential_model(sessions_for_test, item_column, transaction_type_column, target_column)

    item_frequency, pair_frequency = calculate_frequencies(sessions_for_kg, item_column)
    knowledge_graph = construct_knowledge_graph(item_frequency, pair_frequency)
    logging.info(f"Total number of graph nodes: {knowledge_graph.number_of_nodes()}")
    logging.info(f"Total number of graph edges: {knowledge_graph.number_of_edges()}")
    
    kg_nodes = list(knowledge_graph.nodes)

    items_not_in_kg = set(events_for_train[item_column].unique()) - set(kg_nodes)

    item_vocabulary = np.array(kg_nodes + list(items_not_in_kg))
    np.save(savefile_path + item_column, item_vocabulary)
    logging.info(f"Item vocabulary saved to {savefile_path + item_column}")
    walks = construct_random_walks(knowledge_graph, kg_nodes, config.process.num_walks, config.process.num_steps, config.process.p, config.process.q)
    
    node2vec_dataset = generate_dataset_for_node2vec_model(walks, config.process.window_size, config.process.num_negative_samples, len(item_vocabulary))
    tf.data.experimental.save(node2vec_dataset, savefile_path + "node2vec_dataset")
    logging.info(f"Node2Vec dataset saved to {savefile_path + 'node2vec_dataset'}")

    adjacency_matrix = nx.adjacency_matrix(knowledge_graph).tocoo()
    indices = np.mat([adjacency_matrix.row + 1, adjacency_matrix.col + 1]).transpose()
    adj_matrix_sparse = tf.SparseTensor(indices, adjacency_matrix.data, dense_shape=(len(item_vocabulary) + 1, ) * 2)
    serialized_adj_matrix = tf.io.serialize_sparse(adj_matrix_sparse).numpy()[0]
    tf.io.write_file(savefile_path + "adjacency_matrix", serialized_adj_matrix)
    logging.info(f"Adjacency matrix of KG saved to {savefile_path + 'adjacency_matrix'}")
    

    sessions_for_train_ds = tf.data.Dataset.from_tensor_slices(
        {
            user_column: sessions_for_train[user_column],
            transaction_type_column: sessions_for_train[transaction_type_column],
            item_column: sessions_for_train[item_column],
            target_column: sessions_for_train[target_column],
        }
    )
    tf.data.experimental.save(sessions_for_train_ds, savefile_path + "train")
    logging.info(f"Train dataset of KG saved to {savefile_path + 'train'}")

    sessions_for_test_ds = tf.data.Dataset.from_tensor_slices(
        {
            user_column: sessions_for_test[user_column],
            transaction_type_column: sessions_for_test[transaction_type_column],
            item_column: sessions_for_test[item_column],
            target_column: sessions_for_test[target_column],
        }
    )
    tf.data.experimental.save(sessions_for_test_ds, savefile_path + "test")
    logging.info(f"Test dataset of KG saved to {savefile_path + 'test'}")

if __name__ == '__main__':
    process_data()
