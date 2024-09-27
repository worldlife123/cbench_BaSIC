from typing import Dict, List, Union, Any
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing
import functools

import networkx as nx
import scipy.sparse as sp
# import walker

def sliding_window_segments(data : Union[bytes, np.ndarray], segment_length=8) -> List[Union[bytes, np.ndarray]]:
    segments = []
    if isinstance(data, (bytes, str)):
        segments = [data[i:(i+segment_length)] for i in range(len(data)-segment_length)]
    elif isinstance(data, np.ndarray):
        # numpy stride trick
        segments = sliding_window_view(data, (segment_length, ))
    else:
        raise ValueError(f"data {data} not supported")

    return segments

def segment_freq_count(data : Union[bytes, np.ndarray], 
    segment_length=8,
    count_unique_freq=False,
    ) -> Dict[Union[bytes, np.ndarray], int]:
    segments = sliding_window_segments(data, segment_length=segment_length)
    if count_unique_freq:
        return {segment : 1 for segment in set(segments)}
    
    segment_freq_dict = dict()
    for segment in segments:
        # segment = segment.tobytes()
        if segment not in segment_freq_dict:
            segment_freq_dict[segment] = 0
        segment_freq_dict[segment] += 1

    return segment_freq_dict

def segment_conditional_freq_count(data : Union[bytes, np.ndarray], 
    segment_length=8, 
    condition_offset=1,
    count_unique_freq=False, # TODO:
    ) -> Dict[Union[bytes, np.ndarray], int]:
    segments = sliding_window_segments(data, segment_length=segment_length+condition_offset)
    segment_conditional_freq_dict = dict()
    for segment in segments:
        # segment = segment.tobytes()
        segment_cur = segment[:segment_length]
        segment_next = segment[condition_offset:]
        if segment_cur not in segment_conditional_freq_dict:                
            segment_conditional_freq_dict[segment_cur] = dict()
        if segment_next not in segment_conditional_freq_dict[segment_cur]:
            segment_conditional_freq_dict[segment_cur][segment_next] = 0
        segment_conditional_freq_dict[segment_cur][segment_next] += 1

    return segment_conditional_freq_dict

def segment_unique_sample(data : Union[bytes, np.ndarray], segment_length=8) -> Dict[Union[bytes, np.ndarray], int]:
    segments = sliding_window_segments(data, segment_length=segment_length)
    segment_freq_set = set(segments)
    return segment_freq_set

def graph_all_nlength_simple_paths(G : nx.Graph, source : Any, length : int):
    visited = dict.fromkeys([source])
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < length:
            if child in visited:
                continue
            if len(visited) == length - 1:
                yield list(visited) + [child]
            visited[child] = None
            stack.append(iter(G[child]))
        elif len(visited) == length:
            # for target in (set(children) | {child}) - set(visited.keys()):
            #     yield list(visited) + [target]
            stack.pop()
            visited.popitem()

def get_path_between_segment(segment1: str, segment2: str):
    assert len(segment1) == len(segment2)
    # search for connection
    start_idx = 0
    while start_idx < len(segment1):
        if segment1[start_idx:] == segment2[:(len(segment2)-start_idx)]: break
        start_idx += 1
    # construct path
    path = [segment1]
    for idx in range(1, start_idx+1):
        cur_segment = segment1[idx:] + \
            (segment2[(-start_idx):(-start_idx+idx)] if idx != start_idx else segment2[(-start_idx):])
        path.append(cur_segment)

    return path

def generate_graph_best_adj_path(graph : nx.Graph, node_weights : Dict[Any, int], depth_limit : int, start_node : Any):
    best_path = None
    best_path_score = 0
    dfs_nodes = list(nx.dfs_preorder_nodes(graph, start_node, depth_limit=depth_limit))
    for path in nx.all_simple_paths(graph, start_node, dfs_nodes[1:], cutoff=depth_limit):
        current_path_score = sum([node_weights.get(seg, 0) for seg in path])
        if current_path_score > best_path_score:
            best_path_score = current_path_score
            best_path = path
    return best_path

def generate_graph_best_adj_path_mp(graph : nx.Graph, node_weights : Dict[str, int], depth_limit : int, *args, num_workers=8, **kwargs) -> Dict[str, List[str]]:

    worker_obj = functools.partial(generate_graph_best_adj_path, graph, node_weights, depth_limit, *args)
    node_path_dict = {node : None for node in graph.nodes}
    if num_workers > 0:
        with multiprocessing.Pool(num_workers) as pool:
            best_paths = pool.map(worker_obj, node_path_dict.keys())
    else:
        best_paths = [worker_obj(node) for node in node_path_dict.keys()]
    for best_path in best_paths:
        if best_path is not None:
            node_path_dict[best_path[0]] = best_path

    return node_path_dict

def generate_random_paths_sparse_graph(G : nx.Graph, sample_size, path_length=5, index_map=None, start_nodes=None, allow_shorter_path=True, eps=1e-6):
    # Calculate transition probabilities between
    # every pair of vertices according to Eq. (3)
    adj_mat = nx.to_scipy_sparse_matrix(G)
    row_sums = adj_mat.sum(axis=1).astype(np.double)
    inv_row_sums = np.reciprocal(row_sums).reshape(-1, 1)
    inv_row_sums[row_sums == 0] = 0
    inv_row_sums = sp.csr_matrix(inv_row_sums)
    transition_probabilities = adj_mat.multiply(inv_row_sums)

    node_map = np.array(G)
    num_nodes = G.number_of_nodes()

    # only sample from indices that contain edges
    if start_nodes is None:
        valid_node_indices = np.arange(num_nodes) # (row_sums > 0).nonzero()[0]
    else:
        valid_node_indices = []
        for idx, node in node_map.items():
            if node in start_nodes:
                valid_node_indices.append(idx)
    for path_index in range(sample_size):
        # Sample current vertex v = v_i uniformly at random
        node_index = np.random.choice(valid_node_indices)
        node = node_map[node_index]

        # Add v into p_r and add p_r into the path set
        # of v, i.e., P_v
        path = [node]

        # Build the inverted index (P_v) of vertices to paths
        if index_map is not None:
            if node in index_map:
                index_map[node].add(path_index)
            else:
                index_map[node] = {path_index}

        starting_index = node_index
        for _ in range(path_length):
            # Randomly sample a neighbor (v_j) according
            # to transition probabilities from ``node`` (v) to its neighbors
            # NOTE: slow
            # neighbor_index = np.random.choice(
            #     num_nodes, p=transition_probabilities[starting_index].toarray()
            # )
            _, all_neighbor_indices, all_neighbor_probs = sp.find(transition_probabilities[starting_index])
            # all_neighbor_probs = all_neighbor_probs.astype(np.double) / all_neighbor_probs.sum()
            # invalid path
            if len(all_neighbor_indices) == 0:
                if not allow_shorter_path:
                    path = []
                break
            neighbor_index = np.random.choice(
                all_neighbor_indices, p=all_neighbor_probs
            )

            # Set current vertex (v = v_j)
            starting_index = neighbor_index

            # Add v into p_r
            neighbor_node = node_map[neighbor_index]
            path.append(neighbor_node)

            # Add p_r into P_v
            if index_map is not None:
                if neighbor_node in index_map:
                    index_map[neighbor_node].add(path_index)
                else:
                    index_map[neighbor_node] = {path_index}

        if len(path) > 0:
            yield path


# TODO: move to unittest
# if __name__ == "__main__":
#     # ['abcd', 'bcde', 'cdef', 'defg', 'efgs']
#     print(get_path_between_segment("abcd", "efgs"))
#     # ['abcd', 'bcde']
#     print(get_path_between_segment("abcd", "bcde"))
#     # ['abcd', 'bcda', 'cdab', 'dabc']
#     print(get_path_between_segment("abcd", "dabc"))

def dict_training_fastcover(samples : List[bytes], 
    dict_length=32768, max_epoches=None,
    k=250, d=8,
    count_unique_freq=False,
    score_freq_mean=False,
    score_with_offset_epoches=0,
    initial_offset=500,
    use_local_extension=False,
    trim_segment=False,
    # conditional epoch
    kc=250, num_conditional_epoches=0, start_conditional_epoch=0,
    conditional_freq_threshold=1, 
    use_path_growth=False, num_growth_iter=100, kg=8,
    allow_self_loop=True, count_unique_cond_freq=False,
    node_prune_threshold=1, max_iter_per_epoch=1000,
    sample_from_graph=True,
    score_factor_edge=4,
    **kwargs
    ) -> bytes:

    d_segments_all = [sliding_window_segments(sample, segment_length=d) for sample in samples]

    # count freqs
    segment_freqs = []
    if count_unique_freq:
        for d_segments in d_segments_all:
            segment_freqs.append({segment : 1 for segment in set(d_segments)})
    else:
        # segment_freqs = [segment_freq_count(sample, segment_length=d) for sample in samples]
        for d_segments in d_segments_all:
            segment_freq_dict = dict()
            for segment in d_segments:
                # segment = segment.tobytes()
                if segment not in segment_freq_dict:
                    segment_freq_dict[segment] = 0
                segment_freq_dict[segment] += 1
            segment_freqs.append(segment_freq_dict)
    # combine freqs
    global_segment_freqs_dict = dict()
    for segment_freq_dict in segment_freqs:
        for segment, freq in segment_freq_dict.items():
            if segment not in global_segment_freqs_dict:
                global_segment_freqs_dict[segment] = 0
            global_segment_freqs_dict[segment] += freq

    # k-length segment
    # k_segments_all = [sliding_window_segments(sample, segment_length=k) for sample in samples]
    trained_segments = []        
    if num_conditional_epoches != 0:
        # count conditional freqs
        segment_conditional_freqs = []
        for d_segments in d_segments_all:
            segment_conditional_freqs_dict = dict()
            # TODO: maybe start from last as conditional matching starts from nearest?
            for idx in range(len(d_segments)-1):
                # segment = segment.tobytes()
                segment_cur = d_segments[idx]
                segment_next = d_segments[idx+1]
                # filter self loop (not recommended?)
                if not allow_self_loop and segment_cur==segment_next:
                    continue
                if segment_cur not in segment_conditional_freqs_dict:
                    segment_conditional_freqs_dict[segment_cur] = dict()
                    if count_unique_cond_freq:
                        segment_conditional_freqs_dict[segment_cur][segment_next] = 1
                        continue
                else:
                    if count_unique_cond_freq:
                        continue
                if segment_next not in segment_conditional_freqs_dict[segment_cur]:
                    # if count_unique_cond_freq:
                    #     segment_conditional_freqs_dict[segment_cur][segment_next] = 1
                    #     continue
                    # else:
                    segment_conditional_freqs_dict[segment_cur][segment_next] = 0
                segment_conditional_freqs_dict[segment_cur][segment_next] += 1
            segment_conditional_freqs.append(segment_conditional_freqs_dict)
        print("Conditional freqs collected!")
        # combine global conditional freq
        global_segment_conditional_freqs_dict = dict()
        for segment_conditional_freqs_dict in segment_conditional_freqs:
            for segment, cond_freq in segment_conditional_freqs_dict.items():
                if segment not in global_segment_conditional_freqs_dict:
                    global_segment_conditional_freqs_dict[segment] = dict()
                for cond_segment, freq in cond_freq.items():
                    if freq < conditional_freq_threshold: continue
                    if cond_segment not in global_segment_conditional_freqs_dict[segment]:
                        global_segment_conditional_freqs_dict[segment][cond_segment] = 0
                    global_segment_conditional_freqs_dict[segment][cond_segment] += freq
        print("Conditional freqs combined!")
        
    # TODO: wrap as a function
    current_epoch = 0
    current_offset = initial_offset + d
    best_score = 0
    # best_segment = None
    best_segment_sample = None
    while (
            current_epoch < start_conditional_epoch and \
            sum([len(segment) for segment in trained_segments]) < dict_length \
            and (max_epoches is None or current_epoch < max_epoches)
        ):
        # local extension: try extend from the current best segment for extra scores
        if use_local_extension and best_segment_sample is not None:
            best_segment_sample = (best_segment_sample[0], max(0, best_segment_sample[1]-k), max(0, best_segment_sample[1]-k)+k-d)
            # +d-1 for extra scores
            best_segment = samples[best_segment_sample[0]][best_segment_sample[1]:(best_segment_sample[2]+d+d-1)]
            best_score = sum([global_segment_freqs_dict.get(segment, 0) for segment in set(sliding_window_segments(best_segment, segment_length=d))])
            print(f"Local extension score: {best_score}")
        else:
            # best_segment = None
            best_segment_sample = None
            best_score = 0

        # NOTE: unlike zdict, this sliding window only obtains k_length segments 
        # without head and tail (which are shorter than k)
        # calculate segment score
        # for k_segments in k_segments_all:
        #     for k_segment in k_segments:
        #         current_score = 0
        #         for d_segment in sliding_window_segments(k_segment, segment_length=d):
        #             if d_segment in global_segment_freqs_dict:
        #                 current_score += global_segment_freqs_dict[d_segment]
        #         if current_score > best_score:
        #             best_score = current_score
        #             best_segment = k_segment
        
        if current_epoch < score_with_offset_epoches:
            for sample_idx, d_segments in enumerate(d_segments_all):
                for window_idx in range(1, len(d_segments)+k-d):
                    start_idx = max(window_idx-k+d, 0)
                    end_idx = min(window_idx, len(d_segments))
                    segment_length = end_idx - start_idx
                    segment_scores = [global_segment_freqs_dict.get(d_segment, 0) for d_segment in set(d_segments[start_idx:end_idx])]
                    segment_scores_offset = np.array(segment_scores) / (np.log2(np.arange(current_offset, current_offset + len(segment_scores)) + 1) + 1)
                    current_score = np.sum(segment_scores_offset)
                    final_score = current_score / segment_length if score_freq_mean else current_score
                    if final_score > best_score:
                        best_score = final_score
                        best_segment_sample = (sample_idx, start_idx, end_idx)
                        # best_segment = samples[sample_idx][start_idx:(end_idx+d)]
        else:
            # zdict implementation, better optimized!
            for sample_idx, d_segments in enumerate(d_segments_all):
                # d_segments_score = {d_segment : global_segment_freqs_dict.get(d_segment, 0) for d_segment in d_segments}
                current_score = 0
                segment_freqs_dict = dict()
                for d_segment_idx, d_segment in enumerate(d_segments):
                    if d_segment not in segment_freqs_dict:
                        segment_freqs_dict[d_segment] = 0
                        if d_segment in global_segment_freqs_dict:
                            segment_score = global_segment_freqs_dict[d_segment]
                            current_score +=  segment_score
                    segment_freqs_dict[d_segment] += 1
                    # remove out-of-k-window segment
                    k_segment_idx = max(d_segment_idx-k+d, 0)
                    segment_length = d_segment_idx + d - k_segment_idx
                    if d_segment_idx > k-d:
                        delete_segment = d_segments[k_segment_idx]
                        segment_freqs_dict[delete_segment] -= 1
                        if segment_freqs_dict[delete_segment] == 0:
                            if delete_segment in global_segment_freqs_dict:
                                segment_score = global_segment_freqs_dict[delete_segment]
                                current_score -= segment_score
                            segment_freqs_dict.pop(delete_segment)
                    
                    final_score = current_score / segment_length if score_freq_mean else current_score
                    if final_score > best_score:
                        best_score = final_score
                        best_segment_sample = (sample_idx, k_segment_idx, d_segment_idx)
                        # best_segment = samples[sample_idx][k_segment_idx:(d_segment_idx+d)]

        # check best segment
        if best_segment_sample is None:
            # no segment selected
            break
        else:
            # NOTE: try to trim the best_segment (seems to degrade)
            if trim_segment:
                best_segment_path = d_segments_all[best_segment_sample[0]][best_segment_sample[1]:best_segment_sample[2]]
                # remove repeating segments using hashdict
                best_segment_path = list({seg: None for seg in best_segment_path}.keys())
                best_segment = b''.join([node[:1] for node in best_segment_path[:-1]] + [best_segment_path[-1], ])
            else:
                best_segment = samples[best_segment_sample[0]][best_segment_sample[1]:(best_segment_sample[2]+d)]
            print(f"Epoch {current_epoch}: Append {len(best_segment)} segment with score {best_score} to dict")
            trained_segments.append(best_segment)
            current_offset += len(best_segment)
            # clear freq from select segments
            for d_segment in sliding_window_segments(best_segment, segment_length=d):
                if d_segment in global_segment_freqs_dict:
                    # global_segment_freqs_dict[d_segment] = 0
                    global_segment_freqs_dict.pop(d_segment)
        
        current_epoch += 1


    current_conditional_epoch = 0
    while (
            (num_conditional_epoches < 0 or current_conditional_epoch < num_conditional_epoches) and \
            sum([len(segment) for segment in trained_segments]) < dict_length \
            and (max_epoches is None or current_epoch < max_epoches)
        ):
        # graph = nx.DiGraph(global_segment_conditional_freqs_dict)
        graph = nx.DiGraph()
        graph.add_nodes_from(global_segment_freqs_dict)
        graph_forward = nx.DiGraph()
        graph_forward.add_nodes_from(global_segment_freqs_dict)
        graph_forward_weighted = nx.DiGraph()
        graph_forward_weighted.add_nodes_from(global_segment_freqs_dict)

        graph_nodes = list(graph.nodes)
        graph_node_weights = np.array([global_segment_freqs_dict[node] for node in graph.nodes])
        weight_normalizer = graph_node_weights.max()
        graph.add_weighted_edges_from(
            # inverse edges
            # ((v, u, np.exp(-data/weight_normalizer)) for u, nbrs in global_segment_conditional_freqs_dict.items() for v, data in nbrs.items())
            ((v, u, data) for u, nbrs in global_segment_conditional_freqs_dict.items() for v, data in nbrs.items())
        )
        graph_forward.add_weighted_edges_from(
            # inverse edges
            # ((v, u, np.exp(-data/weight_normalizer)) for u, nbrs in global_segment_conditional_freqs_dict.items() for v, data in nbrs.items())
            ((u, v, data) for u, nbrs in global_segment_conditional_freqs_dict.items() for v, data in nbrs.items())
        )
        graph_forward_weighted.add_weighted_edges_from(
            ((u, v, max(0, data * score_factor_edge + global_segment_freqs_dict.get(v, 0))) for u, nbrs in global_segment_conditional_freqs_dict.items() for v, data in nbrs.items())
        )
        print("Graph Updated!")

        if use_path_growth:
            best_path = None
            best_path_score = 0
            # construct an initial path
            # initial path chooses max freq
            for sample_idx, d_segments in enumerate(d_segments_all):
                # d_segments_score = {d_segment : global_segment_freqs_dict.get(d_segment, 0) for d_segment in d_segments}
                current_score = 0
                segment_freqs_dict = dict()
                for d_segment_idx, d_segment in enumerate(d_segments):
                    if d_segment not in segment_freqs_dict:
                        segment_freqs_dict[d_segment] = 0
                        if d_segment in global_segment_freqs_dict:
                            segment_score = global_segment_freqs_dict[d_segment]
                            current_score += segment_score
                    segment_freqs_dict[d_segment] += 1
                    # remove out-of-k-window segment
                    k_segment_idx = max(d_segment_idx-d, 0)
                    segment_length = d_segment_idx + d - k_segment_idx
                    if d_segment_idx >= d:
                        delete_segment = d_segments[k_segment_idx]
                        segment_freqs_dict[delete_segment] -= 1
                        if segment_freqs_dict[delete_segment] == 0:
                            if delete_segment in global_segment_freqs_dict:
                                segment_score = global_segment_freqs_dict[delete_segment]
                                current_score -= segment_score
                            segment_freqs_dict.pop(delete_segment)
                    
                    final_score = current_score
                    if final_score > best_path_score:
                        best_path_score = final_score
                        best_segment = samples[sample_idx][k_segment_idx:(d_segment_idx + d)]
            trained_segments.append(best_segment)
            current_offset += len(best_segment)
            current_segment = best_segment[:d]

            next_segment = max(global_segment_freqs_dict, key=global_segment_freqs_dict.get)
            next_path = get_path_between_segment(current_segment, next_segment)
            next_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in next_path])
            best_path = next_path
            best_path_score = next_path_score
            best_path_updated = False
            cumulate_segment_length = 0

            max_growth_iter = min(num_growth_iter, dict_length // d)
            for iter_idx in range(max_growth_iter):
                print(f"Conditional Graph Growth Iteration {iter_idx}:")

                # NOTE: try to optimize COVER samples by graph growth
                # # pre-calculate all adj paths
                # best_path = None
                # best_path_score = 0
                # node_path_dict = generate_graph_best_adj_path_mp(graph, global_segment_freqs_dict, kg)
                # print("Adj paths generated!")
                # for sample_idx, d_segments in enumerate(d_segments_all):
                #     # d_segments_score = {d_segment : global_segment_freqs_dict.get(d_segment, 0) for d_segment in d_segments}
                #     current_score = 0
                #     segment_freqs_dict = dict()
                #     for d_segment_idx, d_segment in enumerate(d_segments):
                #         if d_segment not in segment_freqs_dict:
                #             segment_freqs_dict[d_segment] = 0
                #             if d_segment in global_segment_freqs_dict:
                #                 segment_score = global_segment_freqs_dict[d_segment]
                #                 current_score += segment_score
                #         segment_freqs_dict[d_segment] += 1
                #         # remove out-of-k-window segment
                #         k_segment_idx = max(d_segment_idx-k+d, 0)
                #         segment_length = d_segment_idx + d - k_segment_idx
                #         if d_segment_idx >= k-d:
                #             delete_segment = d_segments[k_segment_idx]
                #             segment_freqs_dict[delete_segment] -= 1
                #             if segment_freqs_dict[delete_segment] == 0:
                #                 if delete_segment in global_segment_freqs_dict:
                #                     segment_score = global_segment_freqs_dict[delete_segment]
                #                     current_score -= segment_score
                #                 segment_freqs_dict.pop(delete_segment)

                #         # optimize last kg segment
                #         kg_segment_idx = k_segment_idx + kg
                #         best_kg_path = d_segments[k_segment_idx:kg_segment_idx]
                #         kg_segment =  d_segments[kg_segment_idx-1]
                #         kg_segment_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in best_kg_path])
                #         best_kg_path_score = kg_segment_score
                #         if node_path_dict.get(kg_segment) is not None:
                #             best_kg_path = node_path_dict[kg_segment]
                #             best_kg_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in best_kg_path])
                #             opt_score = best_kg_path_score + current_score - kg_segment_score
                #             if opt_score > best_path_score:
                #                 best_path_score = opt_score
                #                 best_path = best_kg_path[::-1] + d_segments[kg_segment_idx:d_segment_idx]
                #                 print(f"Find opt {opt_score} segment instead of {current_score}")
                #                 # best_segment = samples[sample_idx][k_segment_idx:(d_segment_idx+d)]
                #         else:
                #             if current_score > best_path_score:
                #                 best_path_score = current_score
                #                 best_path = d_segments[k_segment_idx:d_segment_idx]

                # if best_path is not None:
                #     # remove duplicated
                #     best_path = list({seg: None for seg in best_path}.keys())
                #     best_segment = b''.join([node[:1] for node in best_path[:-1]] + [best_path[-1], ])
                #     trained_segments.append(best_segment)
                #     print(f"Append {len(best_segment)} segment with path score {best_path_score} to dict")
                #     # clear freq from select segments
                #     for node in best_path:
                #         if node in global_segment_freqs_dict:
                #             # global_segment_freqs_dict[d_segment] = 0
                #             global_segment_freqs_dict.pop(node)
                #         # if node in global_segment_conditional_freqs_dict:
                #         #     global_segment_conditional_freqs_dict.pop(node)
                #         # for segment, cond_freq_dict in global_segment_conditional_freqs_dict.items():
                #         #     if node in cond_freq_dict:
                #         #         cond_freq_dict.pop(node)
                # else:
                #     print("Conditional Epoch found no paths! Skipping...")
                #     break


                dfs_nodes = list(nx.dfs_preorder_nodes(graph, current_segment, depth_limit=d))
                for path in nx.all_simple_paths(graph, current_segment, dfs_nodes[1:], cutoff=d):
                    # if len(path) < d: continue # filter out non-kmer paths
                    # current_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in path])
                    current_path_score = global_segment_freqs_dict.get(path[0], 0)
                    for idx in range(len(path)-1):
                        # current_path_score += global_segment_freqs_dict[path[idx+1]] - global_segment_conditional_freqs_dict[path[idx]][path[idx+1]]
                        # inverse path
                        current_path_score += global_segment_freqs_dict.get(path[idx+1], 0) + global_segment_conditional_freqs_dict[path[idx+1]].get(path[idx], 0)
                    # print(f"Path: {path}")
                    # print(f"Score: {current_path_score}")
                    if current_path_score > best_path_score:
                        best_path_score = current_path_score
                        best_path = path
                        best_path_updated = True

                if best_path is not None:
                    best_segment = best_path[-1]
                    current_segment = best_segment
                    # print(f"    Best segment is {best_segment}.")
                    # print(f"    Best path is {best_path}")
                    # a full transition, append it
                    if len(best_path) == d:
                        append_segment = current_segment
                    # a part transition, append it to the last segment
                    else:
                        if len(best_path) > 1:
                            append_segment = current_segment[-(len(best_path)-1):]
                        else:
                            raise ValueError("The path is not updated!")
                    trained_segments.append(append_segment)
                    current_offset += len(best_segment)
                    cumulate_segment_length += len(append_segment)
                    if not best_path_updated:
                        print("    Starting new path!")
                    print(f"    Append {append_segment} segment with path score {best_path_score} to dict")
                    # clear freq from select segments
                    for idx, node in enumerate(best_path):
                        if node in global_segment_freqs_dict:
                            # global_segment_freqs_dict[d_segment] = 0
                            global_segment_freqs_dict.pop(node)
                        # if idx < len(best_path) - 1 and graph.has_node(node): # leave the last node as we need it in the next iteration
                        #     graph.remove_node(node)
                        # if node in global_segment_conditional_freqs_dict:
                        #     global_segment_conditional_freqs_dict.pop(node)
                        # for segment, cond_freq_dict in global_segment_conditional_freqs_dict.items():
                        #     if node in cond_freq_dict:
                        #         cond_freq_dict.pop(node)
                
                if cumulate_segment_length >= k or best_path is None:
                    # update next best path
                    next_segment = max(global_segment_freqs_dict, key=global_segment_freqs_dict.get)
                    next_path = get_path_between_segment(current_segment, next_segment)
                    next_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in next_path])
                    # print(f"    Next segment {next_segment} with path score {next_path_score}.")
                    best_path = next_path
                    best_path_score = next_path_score
                    cumulate_segment_length = 0
                else:
                    best_path = None
                    best_path_score = 0
                best_path_updated = False
        else:
                # iterate by finding best path
                # # prune graph to a DAG (too slow)
                # pruned_edges = []
                # for cycle in nx.simple_cycles(graph):
                #     if len(cycle) == 1:
                #         # self loop, just remove
                #         pruned_edges.append((cycle[0], cycle[0]))
                #     else:
                #         edge_weights = [graph.get_edge_data(cycle[idx], cycle[idx+1])["weight"] for idx in range(len(cycle)-1)]
                #         best_prune_idx = np.argmin(edge_weights)
                #         pruned_edges.append((cycle[best_prune_idx], cycle[best_prune_idx+1]))
                # for edge in pruned_edges:
                #     graph.remove_edge(edge[0], edge[1])

                # # recursive matrix multiplication (suffers from cycles)
                # graph_weighted_adj_exp = nx.to_scipy_sparse_matrix(graph, format="csr")
                # nodes_weight_adj_exp = sp.diags(np.exp(graph_node_weights/weight_normalizer), format="csr")
                # current_weighted_adj_exp = nodes_weight_adj_exp
                # current_adj = sp.eye(len(graph_node_weights))
                # graph_adj_noweight = graph_weighted_adj_exp > 0
                # num_steps = 0
                # for num_steps in range(kc-d):
                #     if current_weighted_adj_exp.nnz > 1e8: break # max 100Mbytes
                #     current_weighted_adj_exp = np.dot(np.dot(current_weighted_adj_exp, graph_weighted_adj_exp), nodes_weight_adj_exp)
                #     current_adj = np.dot(current_adj, graph_adj_noweight)
                # find best (kc-d)-length path
                # mean_adj_exp = current_weighted_adj_exp
                # mean_adj_exp[current_weighted_adj_exp>0] = current_weighted_adj_exp[current_weighted_adj_exp>0] / current_adj[current_weighted_adj_exp>0]
                # max_path_idx = mean_adj_exp.argmax()
                # max_path_s = max_path_idx // len(graph_node_weights)
                # max_path_t = max_path_idx % len(graph_node_weights)
                # all_possible_node_st = np.nonzero(current_weighted_adj_exp >= np.max(current_weighted_adj_exp[current_weighted_adj_exp>0] / current_adj[current_weighted_adj_exp>0]))
                # num_node_st = len(all_possible_node_st[0])
                # print(f"Find {num_node_st} possible node st! Iterating...")

                # select a path
                best_path = None
                best_path_score = 0
                # combined_weight_graph = nx.DiGraph()
                # combined_weight_graph.add_nodes_from(graph_nodes)
                # combined_weight_graph.add_weighted_edges_from(
                #     ((u, v, global_segment_freqs_dict.get(u, 0) + score_factor_edge * data) for u, nbrs in global_segment_conditional_freqs_dict.items() for v, data in nbrs.items())
                # )
                # if node_prune_threshold > 0:
                #     nodes_to_prune = [k for k, v in global_segment_freqs_dict.items() if v<node_prune_threshold]
                #     combined_weight_graph.remove_nodes_from(nodes_to_prune)
                num_steps = kc-d
                print("Finding best path...")

                # start a dfs from the nodes with highest weight
                # num_iters = 0
                # all_possible_node_st = []
                # for comp in nx.weakly_connected_components(combined_weight_graph):
                #     if len(comp) < kc-d: continue # leave out small comps
                #     comp_nodes = list(comp)
                #     node_weights = [global_segment_freqs_dict[node] for node in comp_nodes]
                #     best_start_node = comp_nodes[np.argmax(node_weights)]
                #     dfs_nodes = list(nx.dfs_preorder_nodes(combined_weight_graph, best_start_node, depth_limit=num_steps))
                #     for dfs_node in dfs_nodes:
                #         if dfs_node != best_start_node:
                #             all_possible_node_st.append((best_start_node, dfs_node)) 
                #     best_start_node_idx = list(combined_weight_graph.nodes).index(best_start_node)
                #     rw_paths = walker.random_walks(combined_weight_graph, n_walks=max_iter_per_epoch, walk_len=num_steps, start_nodes=[best_start_node_idx])
                #     for path in nx.all_simple_paths(combined_weight_graph, best_start_node, dfs_nodes[1:], cutoff=num_steps):
                #     # for path in graph_all_nlength_simple_paths(combined_weight_graph, best_start_node, num_steps):
                #         if len(path) < num_steps: continue # filter out non-kmer paths
                #         current_path_score = global_segment_freqs_dict[path[0]]
                #         for idx in range(len(path)-1):
                #             # current_path_score += global_segment_freqs_dict[path[idx+1]] - global_segment_conditional_freqs_dict[path[idx]][path[idx+1]]
                #             # inverse path
                #             current_path_score += global_segment_freqs_dict[path[idx+1]] - global_segment_conditional_freqs_dict[path[idx+1]][path[idx]]
                #         if current_path_score > best_path_score:
                #             best_path_score = current_path_score
                #             best_path = path
                #         num_iters += 1
                #         print(f"[{num_iters}/{max_iter_per_epoch}] Current path length is {len(path)}, Current best score is {best_path_score}")
                #         if num_iters >= max_iter_per_epoch: break
                
                # best_path = nx.shortest_path(combined_weight_graph, graph_nodes[max_path_s], graph_nodes[max_path_t])
                # for step, (max_path_s, max_path_t) in enumerate(zip(all_possible_node_st[0], all_possible_node_st[1])):
                #     for path in nx.all_simple_paths(graph, graph_nodes[max_path_s], graph_nodes[max_path_t], cutoff=num_steps+1):
                # num_node_st = len(all_possible_node_st)
                # for step, (max_path_s, max_path_t) in enumerate(all_possible_node_st):
                #     print(f"Searching simple paths between {max_path_s} and {max_path_t}")
                #     for path in nx.all_simple_paths(combined_weight_graph, max_path_s, max_path_t, cutoff=num_steps):
                #         current_path_score = global_segment_freqs_dict[path[0]]
                #         for idx in range(len(path)-1):
                #             current_path_score += global_segment_freqs_dict[path[idx+1]] - global_segment_conditional_freqs_dict[path[idx]][path[idx+1]]
                #         if current_path_score > best_path_score:
                #             best_path_score = current_path_score
                #             best_path = path
                #     print(f"[{step}/{num_node_st}] Current path length is {len(path)}, Current best score is {best_path_score}")

                # TODO: optimize segment score counting! refer to zdict implementation!
                # get samples from original data
                for sample_idx, d_segments in enumerate(d_segments_all):
                    for d_segment_idx, d_segment in enumerate(d_segments):
                        if (d_segment_idx+num_steps) > len(d_segments): continue
                        path = d_segments[d_segment_idx:(d_segment_idx+num_steps+1)]
                        # current_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in path])
                        # NOTE: count non-overlapping score
                        # current_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in set(path)])
                        # NOTE: conditional score
                        current_path_score = 0
                        counted_segments = set()
                        for idx in range(len(path)-1, 0, -1):
                            if path[idx] in counted_segments:
                                continue
                            counted_segments.add(path[idx])
                            current_path_score += global_segment_freqs_dict.get(path[idx], 0)
                            if path[idx-1] in global_segment_conditional_freqs_dict:
                                current_path_score += score_factor_edge * global_segment_conditional_freqs_dict[path[idx-1]].get(path[idx], 0)
                            # inverse path
                            # current_path_score += global_segment_freqs_dict.get(path[idx], 0) - global_segment_conditional_freqs_dict[path[idx]].get(path[idx-1], 0)
                        # last segment
                        if not path[0] in counted_segments:
                            current_path_score += global_segment_freqs_dict.get(path[0], 0)
                        if current_path_score > best_path_score:
                            best_path_score = current_path_score
                            best_path = path

                # generate random samples from graph
                if sample_from_graph:
                    for path_idx, path in enumerate(generate_random_paths_sparse_graph(graph_forward_weighted, max_iter_per_epoch, path_length=num_steps)):
                        if len(path) < num_steps: continue # filter out non-kmer paths
                        # current_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in path])
                        # NOTE: count non-overlapping score
                        # current_path_score = sum([global_segment_freqs_dict.get(seg, 0) for seg in set(path)])
                        # NOTE: conditional score
                        current_path_score = 0
                        counted_segments = set()
                        for idx in range(len(path)-1, 0, -1):
                            if path[idx] in counted_segments:
                                continue
                            counted_segments.add(path[idx])
                            current_path_score += global_segment_freqs_dict.get(path[idx], 0)
                            if path[idx-1] in global_segment_conditional_freqs_dict:
                                current_path_score += score_factor_edge * global_segment_conditional_freqs_dict[path[idx-1]].get(path[idx], 0)
                            # inverse path
                            # current_path_score += global_segment_freqs_dict.get(path[idx], 0) - global_segment_conditional_freqs_dict[path[idx]].get(path[idx-1], 0)
                        # last segment
                        if not path[0] in counted_segments:
                            current_path_score += global_segment_freqs_dict.get(path[0], 0)
                        print(f"Sample {path_idx}/{max_iter_per_epoch} score {current_path_score}, best score {best_path_score}")
                        if current_path_score > best_path_score:
                            best_path_score = current_path_score
                            best_path = path

                # append best path and remove freqs from dict
                if best_path is not None:
                    # best_segment = b''.join([best_path[0], ] + [node[-1:] for node in best_path[1:]])
                    # inverse path
                    best_segment = b''.join([node[:1] for node in best_path[:-1]] + [best_path[-1], ])
                    trained_segments.append(best_segment)
                    current_offset += len(best_segment)
                    print(f"Conditional Epoch {current_epoch}: Append {len(best_segment)} segment with path score {best_path_score} to dict")
                    # clear freq of select segments
                    for node in best_path:
                        if node in global_segment_freqs_dict:
                            # global_segment_freqs_dict[d_segment] = 0
                            global_segment_freqs_dict.pop(node)
                        # if node in global_segment_conditional_freqs_dict:
                        #     global_segment_conditional_freqs_dict.pop(node)
                    # clear freq of select segments from conditional dict
                    for idx in range(len(best_path)-1):
                        if best_path[idx] in global_segment_conditional_freqs_dict:
                            global_segment_conditional_freqs_dict.pop(best_path[idx])
                            # NOTE: pop edge?
                            # cond_freq_dict = global_segment_conditional_freqs_dict[best_path[idx]]
                            # if best_path[idx+1] in cond_freq_dict:
                            #     cond_freq_dict.pop(best_path[idx+1])
                        # for segment, cond_freq_dict in global_segment_conditional_freqs_dict.items():
                        #     if node in cond_freq_dict:
                        #         cond_freq_dict.pop(node)
                else:
                    print("Conditional Epoch found no paths! Skipping...")
                    break
        
        current_conditional_epoch += 1
        current_epoch += 1

    # find best k-mer iteratively
    # current_epoch = 0
    # current_offset = initial_offset + d
    best_score = 0
    # best_segment = None
    best_segment_sample = None
    while (
            sum([len(segment) for segment in trained_segments]) < dict_length
            and (max_epoches is None or current_epoch < max_epoches)
        ):
        # local extension: try extend from the current best segment for extra scores
        if use_local_extension and best_segment_sample is not None:
            best_segment_sample = (best_segment_sample[0], max(0, best_segment_sample[1]-k), max(0, best_segment_sample[1]-k)+k-d)
            # +d-1 for extra scores
            best_segment = samples[best_segment_sample[0]][best_segment_sample[1]:(best_segment_sample[2]+d+d-1)]
            best_score = sum([global_segment_freqs_dict.get(segment, 0) for segment in set(sliding_window_segments(best_segment, segment_length=d))])
            print(f"Local extension score: {best_score}")
        else:
            # best_segment = None
            best_segment_sample = None
            best_score = 0

        # NOTE: unlike zdict, this sliding window only obtains k_length segments 
        # without head and tail (which are shorter than k)
        # calculate segment score
        # for k_segments in k_segments_all:
        #     for k_segment in k_segments:
        #         current_score = 0
        #         for d_segment in sliding_window_segments(k_segment, segment_length=d):
        #             if d_segment in global_segment_freqs_dict:
        #                 current_score += global_segment_freqs_dict[d_segment]
        #         if current_score > best_score:
        #             best_score = current_score
        #             best_segment = k_segment
        
        if current_epoch < score_with_offset_epoches:
            for sample_idx, d_segments in enumerate(d_segments_all):
                for window_idx in range(1, len(d_segments)+k-d):
                    start_idx = max(window_idx-k+d, 0)
                    end_idx = min(window_idx, len(d_segments))
                    segment_length = end_idx - start_idx
                    segment_scores = [global_segment_freqs_dict.get(d_segment, 0) for d_segment in set(d_segments[start_idx:end_idx])]
                    segment_scores_offset = np.array(segment_scores) / (np.log2(np.arange(current_offset, current_offset + len(segment_scores)) + 1) + 1)
                    current_score = np.sum(segment_scores_offset)
                    final_score = current_score / segment_length if score_freq_mean else current_score
                    if final_score > best_score:
                        best_score = final_score
                        best_segment_sample = (sample_idx, start_idx, end_idx)
                        # best_segment = samples[sample_idx][start_idx:(end_idx+d)]
        else:
            # zdict implementation, better optimized!
            for sample_idx, d_segments in enumerate(d_segments_all):
                # d_segments_score = {d_segment : global_segment_freqs_dict.get(d_segment, 0) for d_segment in d_segments}
                current_score = 0
                segment_freqs_dict = dict()
                for d_segment_idx, d_segment in enumerate(d_segments):
                    if d_segment not in segment_freqs_dict:
                        segment_freqs_dict[d_segment] = 0
                        if d_segment in global_segment_freqs_dict:
                            segment_score = global_segment_freqs_dict[d_segment]
                            current_score += segment_score
                    segment_freqs_dict[d_segment] += 1
                    # remove out-of-k-window segment
                    k_segment_idx = max(d_segment_idx-k+d, 0)
                    segment_length = d_segment_idx + d - k_segment_idx
                    if d_segment_idx > k-d:
                        delete_segment = d_segments[k_segment_idx]
                        segment_freqs_dict[delete_segment] -= 1
                        if segment_freqs_dict[delete_segment] == 0:
                            if delete_segment in global_segment_freqs_dict:
                                segment_score = global_segment_freqs_dict[delete_segment]
                                current_score -= segment_score
                            segment_freqs_dict.pop(delete_segment)
                    
                    final_score = current_score / segment_length if score_freq_mean else current_score
                    if final_score > best_score:
                        best_score = final_score
                        best_segment_sample = (sample_idx, k_segment_idx, d_segment_idx)
                        # best_segment = samples[sample_idx][k_segment_idx:(d_segment_idx+d)]

        # check best segment
        if best_segment_sample is None:
            # no segment selected
            break
        else:
            # NOTE: try to trim the best_segment (seems to degrade)
            if trim_segment:
                best_segment_path = d_segments_all[best_segment_sample[0]][best_segment_sample[1]:best_segment_sample[2]]
                # remove repeating segments using hashdict
                best_segment_path = list({seg: None for seg in best_segment_path}.keys())
                best_segment = b''.join([node[:1] for node in best_segment_path[:-1]] + [best_segment_path[-1], ])
            else:
                best_segment = samples[best_segment_sample[0]][best_segment_sample[1]:(best_segment_sample[2]+d)]
            print(f"Epoch {current_epoch}: Append {len(best_segment)} segment with score {best_score} to dict")
            trained_segments.append(best_segment)
            current_offset += len(best_segment)
            # clear freq from select segments
            for d_segment in sliding_window_segments(best_segment, segment_length=d):
                if d_segment in global_segment_freqs_dict:
                    # global_segment_freqs_dict[d_segment] = 0
                    global_segment_freqs_dict.pop(d_segment)
        
        current_epoch += 1

    # reverse to allow segments with higher score to be assigned lower offset
    total_dict = b''.join(reversed(trained_segments))
    if len(total_dict) > dict_length:
        total_dict = total_dict[(-dict_length):]
    return total_dict


def dict_training_fastcover_tryparameters(samples : List[bytes], 
    dict_length=32768, max_epoches=None,
    kmin=50, kmax=2000, kstep=50,
    dmin=6, dmax=8, dstep=2,
    **kwargs
    ):
    trained_dict_all = []
    # try compression using all dicts and use the best one
    for k in range(kmin, kmax+kstep, kstep):
        for d in range(dmin, dmax+dstep, dstep):
            dict = dict_training_fastcover(samples, dict_length=dict_length, max_epoches=max_epoches, k=k, d=d, **kwargs)
            trained_dict_all.append(dict)

    # TODO: try compression?
