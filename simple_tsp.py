import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# from problems.stsp.problem_stsp import generate_instance
from utils import load_model

def tsp_solver(model, info, file, distance_matrix=None):


    model.eval()  # Put in evaluation mode to not track gradients

    def make_oracle(model, info, temperature=1.0):
        xy = info['orders_coords_std']
        num_nodes = len(xy)
        info['orders_coords_std']=info['orders_coords_std'].unsqueeze(dim=0)
        # xyt = torch.tensor(xy).float()[None]  # Add batch dimension

        with torch.no_grad():  # Inference only
            embeddings, _ = model.embedder(model._init_embed(info))
            # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
            fixed = model._precompute(embeddings)

        def oracle(tour):
            with torch.no_grad():  # Inference only
                # Input tour with 0 based indices
                # Output vector with probabilities for locations not in tour
                tour = torch.tensor(tour).long()
                if len(tour) == 0:
                    step_context = model.W_placeholder
                else:
                    step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

                # Compute query = context node embedding, add batch and step dimensions (both 1)
                query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

                # Create the mask and convert to bool depending on PyTorch version
                mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
                mask[tour] = 1
                mask = mask[None, None, :]  # Add batch and step dimension

                log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
                p = torch.softmax(log_p / temperature, -1)[0, 0]
                assert (p[tour] == 0).all()
                assert (p.sum() - 1).abs() < 1e-5
                # assert np.allclose(p.sum().item(), 1)
            return p.tolist()

        return oracle

    oracle = make_oracle(model, info)

    xy = info['order_after_delete']  # 随机抽取的点的idx(pad)
    tour = []

    while (len(tour) < len(xy)):
        p = oracle(tour)
        p = np.array(p)
        i = np.argmax(p)
        tour.append(i)  # idx的idx
        # i是重新编号之后的了 但是neighbor_nodes用的还是原来的idx
        neighbor_nodes = info['neighbor_nodes']
        if info['order_after_delete'][i].item() in neighbor_nodes:
            try:
                if info['order_after_delete'].tolist().index(neighbor_nodes[info['order_after_delete'][i].item()]) not in tour:
                    # 这里是取到邻接节点重新编号后的idx
                    tour.append(info['order_after_delete'].tolist().index(neighbor_nodes[info['order_after_delete'][i].item()]))
            except:
                a=1
    tour.append(tour[0])
    tensor_tour = torch.tensor(tour, device='cuda')
    d = info['order_after_delete'].unsqueeze(0).gather(1, tensor_tour.unsqueeze(0))
    print(d)
    required_info = np.load(f'files/{file}/required_info.npy', allow_pickle=True)
    coords_before_delete_x = required_info[1]
    coords_before_delete_y = required_info[2]
    d_np = d.cpu().numpy().flatten()
    coords_before_delete_of_d_x = coords_before_delete_x[d_np]
    coords_before_delete_of_d_y = coords_before_delete_y[d_np]
    # combined_xy = [(a, b) for a, b in zip(coords_before_delete_of_d_x, coords_before_delete_of_d_y)]

    next_d = d[:, 1:]
    prev_d = d[:, :-1]
    if distance_matrix is None:
        distance_matrix = np.load(f'files/{file}/distance_matrix.npy', allow_pickle=True)
    l = 0
    prev_d_item = prev_d[0].tolist()
    next_d_item = next_d[0].tolist()
    for i in range(len(prev_d_item)):
        distance = distance_matrix[prev_d_item[i]][next_d_item[i]]
        l += distance
    print(l)
    return l, coords_before_delete_of_d_x, coords_before_delete_of_d_y

