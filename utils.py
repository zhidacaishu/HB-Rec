import torch 
import numpy as np
import math
from collections import Counter
import pandas as pd
import itertools
from tqdm import tqdm

def calculate_CR(
    recommendations_for_all_users, 
    candidate_items_of_all_users
):
    """
    Calculates the Cumulative Recall when N items are recommended to the user

    Parameters
    ----------
    recommendations_for_all_users: list of lists, shape [num_users, num_items]
        A list of the top N items list recommended to each user

    candidate_items_of_all_users: list of lists, shape [num_users, num_items]
        Each sublist is a list of items that the user actually selected as the first element, 
        and all remaining elements are a list of negatively sampled items

    Returns
    -------
    CR: flaot
        Cumulative Recall obtained by the final calculation
    """
    hit_list = []

    for recommended_items, candidate_data in zip(recommendations_for_all_users, candidate_items_of_all_users):
        hit = int(candidate_data[0] in recommended_items)
        hit_list.append(hit)
    
    CR = sum(hit_list) / len(hit_list) if hit_list else 0

    return CR

def calculate_NDCG(
    recommendations_for_all_users,
    candidate_items_of_all_users
):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) for a list of recommendations and true item.

    Parameters
    ----------
    recommendations_for_all_users: list of lists, shape [num_users, top_n_recommendation]
        A list containing recommendations for each user. Each sublist contains recommended item indices.
    
    candidate_items_of_all_users: list of lists, shape [num_users, num_items]
        Each sublist is a list of items that the user actually selected as the first element, 
        and all remaining elements are a list of negatively sampled items

    Returns
    -------
    NDCG: float
        Normalized Discounted Cumulative Gain (NDCG) value calculated based on the recommendations and true items.
    """
    term_list = []
    for recommended_items, candidate_data in zip(recommendations_for_all_users, candidate_items_of_all_users):
        hit = int(candidate_data[0] in recommended_items)
        term_list.append((1 / math.log2(recommended_items.index(candidate_data[0]) + 2)) if hit else 0)

    NDCG = sum(term_list) / len(term_list) if term_list else 0

    return NDCG

def calculate_topic_diversity(
    topic_word_dist, 
    top_n
):
    """
    Calculate the degree of differentiation between topics

    Parameters
    ----------
    topic_word_dist: torch.Tensor, shape [num_topics, vocab_size]
        A tensor containing the word distribution for each topic. 
        Each row represents a topic and contains the probabilities of each word in the vocabulary.
    
    top_n: int
        The number of top words to consider from each topic's word distribution for calculating diversity.

    Returns
    -------
    topic_diversity: list, shape [num_topics]
        A list containing the diversity score for each topic. 
        Each value represents the proportion of unique top_n words that are not shared with other topics for that topic.
    
    topic_diversity_mean: float
        The mean of the diversity scores across all topics.
    
    topic_diversity_variance: float
        The variance of the diversity scores across all topics.
    """
    topics, vocab_size = topic_word_dist.size()
    topk, indices = torch.topk(topic_word_dist, top_n, dim=1)
    
    unique_indices = []
    for k in range(topics):
        top_indices = indices[k]
        is_unique = torch.ones(top_n, dtype=torch.bool, device=topic_word_dist.device)

        for other_k in range(topics):
            if other_k != k:
                is_unique = is_unique & (~top_indices.unsqueeze(1).eq(indices[other_k]).any(dim=1))
        unique_indices.append(is_unique.sum().float() / top_n)

    topic_diversity = unique_indices
    diversity_mean = torch.mean(topic_diversity)
    diversity_variance = torch.var(topic_diversity)

    return topic_diversity, diversity_mean.item(), diversity_variance.item()

def recommend_N(
    item_probability_matrix, 
    candidate_list_of_all_users, 
    top_n
):
    """
    To recommend top_n items of candidate list for all users

    Parameters
    ----------
    item_probability_matrix: torch.Tensor, shape [num_users, num_items]
        A tensor containing the probability of each user selecting each item.
        Each row represents a user and each column represents an item.
    
    candidate_list_of_all_users: list of lists, length num_users
        A list where each element is a list of candidate item indices for the corresponding user.
    
    top_n: int
        The number of top items to recommend for each user from their candidate list.
    
    Returns
    -------
    recommendations: list of lists, length num_users
        A list where each element is a list of top_n recommended item indices for the corresponding user.
    """
    num_users = item_probability_matrix.size(0)
    recommendations = []

    for user_id in range(num_users):
        candidate_list = candidate_list_of_all_users[user_id]

        candidate_probabilities = item_probability_matrix[user_id, candidate_list]
        
        _, top_indices = torch.topk(candidate_probabilities, top_n)
        
        recommended_items = [candidate_list[i] for i in top_indices.tolist()]
        
        recommendations.append(recommended_items)
    
    return recommendations

def calculate_popularity(
    track_count_list,
    top_1_recommend_list
):
    """
    To calculate the average popularity of the most likely favorite tracks recommended to all users
    
    Parameters
    ----------
    track_count_list: torch.Tensor, length num_items
        A tensor that records the number of plays of all tracks.

    top_1_recommend_list: list, length num_users
        A list of the most likely favorite tracks recommended to all users.
    
    Returns 
    -------
    popularity: float
        The average popularity of the most likely favorite tracks recommended to all users.
    """
    
    num_users = len(top_1_recommend_list)
    popularity = torch.sum(track_count_list[top_1_recommend_list]) / num_users

    return popularity

def calculate_Gini(
    user_events_list
):
    """
    To calculate the processed Gini index of each user's listening history

    Parameters
    ----------
    user_events_list: list of lists, length num_users
        A list of lists containing all users' listening history 
    
    Returns
    -------
    normalized_h_index: list 
        A list of processed Gini indices
    """
    all_usrs_gini_index = []
    for listened_tracks in user_events_list:
        num_tracks = len(set(listened_tracks))

        counter = Counter(listened_tracks)

        sorted_items = sorted(counter.items())

        _, counts = zip(*sorted_items)

        mean = sum(counts) / len(counts)
        sum_part = 0
        for i in range(num_tracks - 1):
            for j in range(i+1, num_tracks): 
                sum_part += abs(counts[i] - counts[j])

        gini_index = sum_part / (num_tracks * mean)
        all_usrs_gini_index.append(gini_index)

    h_index = torch.log2(1 / (torch.tensor(all_usrs_gini_index) + 1))
    normalized_h_index = h_index / torch.sum(h_index)

    return normalized_h_index.tolist()

def calculate_popular_track_ratio(
    track_count_list,
    user_events_list
):
    """
    To calculate the percentage of popul music in each user's listening history.

    Parameters
    ----------
    track_count_list: list, length num_items
        A list that records the number of plays of all tracks.
    
    user_events_list: list of lists, length num_users
        A list of lists containing all users' listening history 

    Returns
    -------
    user_ratios: list, length num_items
        A list of the percentage of popular music in the listening history of all users.
    """
    sorted_track_counts = sorted(track_count_list, reverse=True)
    num_popular_tracks = int(len(track_count_list) * 0.2)
    popular_threshold = sorted_track_counts[num_popular_tracks - 1]
    popular_tracks = [track for track in range(len(track_count_list)) if track_count_list[track] >= popular_threshold]

    user_ratios = []
    for user_listend_tracks in user_events_list:
        if len(user_listend_tracks) == 0:
            user_ratios.append(0)
        else:
            popular_count = sum(1 for track in user_listend_tracks if track in popular_tracks)
            user_ratios.append(popular_count / len(user_listend_tracks))
    
    return user_ratios 

def calculate_topic_diversity(
    topic_word_distribution, 
    top_n=50
):
    """
    Calculate the diversity of topics based on topic word distributions.

    Parameters
    ----------
    topic_word_distribution: torch.Tensor, shape [num_topics, vocab_size]
        A tensor representing the distribution of words across multiple topics. 

    top_n: int, length top_n 
        The number of top words to consider for each topic when calculating uniqueness.

    Returns
    -------
    tuple: tuple, length 3
        - topic_diversity (torch.Tensor): A tensor containing the uniqueness scores of each topic.
        - diversity_mean (float): The mean uniqueness score across all topics.
        - diversity_variance (float): The variance of the uniqueness scores across all topics.
    """
    num_topics, vocab_size = topic_word_distribution.size()
    topk, indices = torch.topk(topic_word_distribution, top_n, dim=1)

    unique_indices = []

    for topic_idx in range(num_topics):
        top_indices = indices[topic_idx]
        is_unique = torch.ones(top_n, dtype=torch.bool, device=topic_word_distribution.device)

        for other_topic_idx in range(num_topics):
            if other_topic_idx != topic_idx:
                is_unique = is_unique & (~top_indices.unsqueeze(1).eq(indices[other_topic_idx]).any(dim=1))
        
        unique_indices.append(is_unique.sum().float() / top_n)

    topic_diversity = torch.tensor(unique_indices)
    diversity_mean = torch.mean(topic_diversity)
    diversity_variance = torch.var(topic_diversity)

    return topic_diversity, diversity_mean.item(), diversity_variance.item()

def _intersection_size(
    list1, 
    list2, 
    d
):
    """
    Calculate the size of the intersection of the top d elements from two lists.

    Parameters
    ----------
    list1: list
        The first list of elements.
    
    list2: list
        The second list of elements.
    
    d: int
        The number of top elements to consider from each list.

    Returns
    -------
    intersection_length: int
        The size of the intersection of the top d elements from both lists.
    """
    subset1 = set(list1[:d])
    subset2 = set(list2[:d])

    intersection_length = len(subset1.intersection(subset2))
    return intersection_length

def _rbo(
    list1, 
    list2, 
    p=0.9
):
    """
    Compute the Rank-Biased Overlap (RBO) score between two ranked lists.

    Parameters
    ----------
    list1: list
        The first ranked list.
    
    list2: list
        The second ranked list.
    
    p: float, optional
        The parameter p that controls the discounting factor (default is 0.9).

    Returns
    -------
    rbo_value: float
        The RBO score between the two ranked lists.
    """
    depth = min(len(list1), len(list2))
    sim = sum([( _intersection_size(list1, list2, d) / d) * (p ** (d - 1)) for d in range(1, depth + 1)])
    rbo_value = (1 - p) * sim
    return rbo_value

def calculate_rbo_mean(
    topic_word_distribution, 
    p=0.9, 
    top_n=50
):
    """
    Calculate the mean RBO scores for each topic based on the top-ranked words.

    Parameters
    ----------
    topic_word_distribution: torch.Tensor, shape [num_topics, vocab_size]
        A tensor representing the distribution of words across multiple topics.
    
    p: float, optional
        The parameter p that controls the discounting factor (default is 0.9).

    top_n: int, length top_n 
        The number of top words to consider for each topic when calculating RBO.

    Returns
    -------
    rbo_means: torch.Tensor, shape [num_topics,]
        The mean RBO scores for each topic.
    """
    num_topics, num_words = topic_word_distribution.shape
    _, top_indices = torch.topk(topic_word_distribution, top_n, dim=1)
    top_indices = top_indices.to('cpu').tolist()

    rbo_matrix = torch.zeros(num_topics, num_topics)

    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            rbo_score = _rbo(top_indices[i], top_indices[j], p)
            rbo_matrix[i, j] = rbo_score
            rbo_matrix[j, i] = rbo_score
    
    rbo_means = sum(torch.sum(rbo_matrix, dim=1) / (num_topics - 1)) / num_topics
    
    return rbo_means

def _all_integer_pairs(
    integer_list
):
    """
    Generate all possible pairs of integers from a list.

    Parameters
    ----------
    integer_list: list
        A list of integers from which pairs are to be generated.

    Returns
    -------
    list of tuples
        A list containing all possible unique pairs of integers.
    """
    return list(itertools.combinations(integer_list, 2))

def _compute_pmi(
    track_x, 
    track_y, 
    user_events_list
):
    """
    Compute the Pointwise Mutual Information (PMI) between two tracks based on user listening events.

    Parameters
    ----------
    track_x: int
        The track ID of the first track.
    
    track_y: int
        The track ID of the second track.
    
    user_events_list: list of lists, length num_users
        A list of lists containing all users' listening history

    Returns
    -------
    torch.Tensor
        The PMI value between the two tracks. Returns 0 if probabilities are zero.
    """
    total_users=len(user_events_list)

    event_sets = [set(events) for events in user_events_list]

    count_x = sum(track_x in events for events in event_sets)
    count_y = sum(track_y in events for events in event_sets)
    count_xy = sum(track_x in events and track_y in events for events in event_sets)

    # count_x = sum(map(lambda sublist: track_x in sublist, user_events_list))
    # count_y = sum(map(lambda sublist: track_y in sublist, user_events_list))
    # count_xy = sum(map(lambda sublist: track_x in sublist and track_y in sublist, user_events_list))

    p_x = count_x / total_users
    p_y = count_y / total_users
    p_xy = count_xy / total_users
    
    if p_xy == 0 or p_x == 0 or p_y == 0:
        return torch.tensor(0.0)
    
    pmi = torch.log(torch.tensor(p_xy / (p_x * p_y)))
    return pmi

def calculate_pmi_mean(
    topic_word_distribution, 
    user_events_list,
    top_n
):
    """
    Calculate the average PMI for each topic based on the top representative tracks.

    Parameters
    ----------
    topic_word_distribution: torch.Tensor, shape [num_topics, vocab_size]
        A tensor representing the distribution of words across multiple topics.

    user_events_list: list of lists, length num_users
        A list of lists containing all users' listening history

    top_n: int
        The number of top n words (tracks) to consider for each topic.

    Returns
    -------
    torch.Tensor, shape (num_topics,)
        The average PMI scores for each topic.
    """
    top_k_indices = torch.topk(topic_word_distribution, top_n, dim=1).indices

    topic_representative_tracks = {}

    for topic_idx, indices in enumerate(top_k_indices):
        topic_representative_tracks[topic_idx] = indices
    
    topic_pmi_scores = torch.zeros(len(topic_word_distribution), device=topic_word_distribution.device)

    for topic_idx in tqdm(range(len(topic_word_distribution)), desc="Calculating PMI"):
        for track_x, track_y in _all_integer_pairs(topic_representative_tracks[topic_idx]):
            topic_pmi_scores[topic_idx] += _compute_pmi(track_x, track_y, user_events_list)
    
    topic_pmi_scores /= (top_n * (top_n - 1)) / 2
    
    return topic_pmi_scores