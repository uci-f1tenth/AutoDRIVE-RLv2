import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def cycle_score(cycle, G):
    avg_depth = np.mean([G.nodes[n]['depth'] for n in cycle])
    length = len(cycle)

    return avg_depth * np.log1p(length)

def step_score(prev, cur, next, G):
    depth = G.nodes[next]['depth']

    if prev is not None:
        v1 = np.array(cur) - np.array(prev)
        v2 = np.array(next) - np.array(cur)
        turn_penalty = 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    else:
        turn_penalty = 0.0

    return depth - 0.3 * turn_penalty

def greedy_order_cycle(G, start):
    ordered = [start]
    visited = {start}
    cur = start

    while True:
        nbrs = list(G.neighbors(cur))

        candidates = []
        for n in nbrs:
            if n not in visited or (n == start and len(ordered) > 10):
                candidates.append(n)

        if not candidates:
            break

        next_node = max(candidates, key=lambda n: G.nodes[n]['depth'])

        if next_node == start:
            ordered.append(start)
            break

        ordered.append(next_node)
        visited.add(next_node)
        cur = next_node

    return ordered

def generate_frenet(map_file_path, output_name = "track"):
    map_image = cv2.imread(map_file_path, cv2.IMREAD_GRAYSCALE)
    if map_image is None:
        raise FileNotFoundError("Map file not found")
    map_image = cv2.copyMakeBorder(map_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0) # adds a black edge to make sure that graph is enclosed

    _, binary_map = cv2.threshold(map_image, 250, 255, cv2.THRESH_BINARY)
    euclidean_dist = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)

    skeleton_map = skeletonize(binary_map > 0)

    y, x = np.where(skeleton_map)
    points = [(int(px), int(py)) for px, py in zip(x, y)]
    point_set = set(points)

    max_depth = np.max(euclidean_dist)
    threshold = max_depth * 0.1 # tune

    filtered_points = [p for p in points if euclidean_dist[p[1], p[0]] > threshold]
    point_set = set(filtered_points)

    G = nx.Graph()
    
    for p in points:            # mapping points
        G.add_node(p)
        G.nodes[p]['depth'] = float(euclidean_dist[p[1], p[0]])
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                neighbor = (p[0] + dx, p[1] + dy)
                if neighbor in point_set:
                    G.add_edge(p, neighbor)

    components = sorted(nx.connected_components(G), key = len, reverse = True)
    if not components:
        raise Exception("No track found in the map")
    max_component = G.subgraph(components[0]).copy()

    while True: # remove degree == 1 (dead end pruning)
        spurs = [node for node, degree in max_component.degree() if degree == 1]
        if not spurs:
            break
        max_component.remove_nodes_from(spurs)

    global_max_depth = max(d for _, d in max_component.nodes(data='depth'))
    soft_min_depth = 0.1 * global_max_depth

    for n, d in list(max_component.nodes(data='depth')):
        if d < soft_min_depth:
            max_component.nodes[n]['depth'] *= 0.2

    components = sorted(nx.connected_components(max_component), key=len, reverse=True)
    if not components:
        raise Exception("No track remains after spur pruning")
    max_component = max_component.subgraph(components[0]).copy()

    for u, v in max_component.edges():
        depth_u = max_component.nodes[u]['depth']
        depth_v = max_component.nodes[v]['depth']
        avg_depth = (depth_u + depth_v) / 2.0
        max_component[u][v]['weight'] = 1.0 / (avg_depth + 1e-6)

    cycles = [
        c for c in nx.cycle_basis(max_component)
        if len(c) > 30
    ]

    if cycles:
        best_cycle = max(cycles, key=lambda c: cycle_score(c, max_component))
        cycle_graph = max_component.subgraph(best_cycle).copy()

        start_node = max(best_cycle, key = lambda n: cycle_graph.nodes[n]['depth'])
        ordered_nodes = greedy_order_cycle(cycle_graph, start_node)

        bc = 'periodic'
        ordered_points = ordered_nodes
        if not np.array_equal(ordered_points[0], ordered_points[-1]):
            ordered_points.append(ordered_points[0])
        # longest_cycle = max(cycles, key = len)
        # cycle_graph = max_component.subgraph(longest_cycle).copy()

        # start_node = max(longest_cycle, key = lambda n: cycle_graph.nodes[n]['depth'])
        # dists = nx.single_source_dijkstra_path_length(cycle_graph, start_node, weight = 'weight')
        # far_node = max(dists, key = dists.get)

        # path_a = nx.shortest_path(cycle_graph, start_node, far_node, weight = 'weight')

        # G_temp = cycle_graph.copy()
        # G_temp.remove_nodes_from(path_a[1:-1])
        # path_b = nx.shortest_path(G_temp, far_node, start_node, weight='weight')
        
        # ordered_points = path_a + path_b[1:]
    else:
        start_node = max(max_component.nodes(), key=lambda n: max_component.nodes[n]['depth'])
        dists = nx.single_source_dijkstra_path_length(max_component, start_node, weight='weight')
        end_node = max(dists, key=dists.get)
        ordered_points = nx.shortest_path(max_component, start_node, end_node, weight='weight')
        bc = 'not-a-knot'

    path = np.array(ordered_points)
    mask = np.ones(len(path), dtype=bool)
    mask[1:] = np.any(np.diff(path, axis=0) != 0, axis=1)
    path = path[mask]

    s_accum = np.zeros(len(path))
    s_accum[1:] = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))

    unique_s_mask = np.diff(s_accum, prepend=-1.0) > 1e-5
    s_accum = s_accum[unique_s_mask]
    path = path[unique_s_mask]

    if bc == 'periodic':
        if not np.allclose(path[0], path[-1]):
            path = np.vstack([path, path[0]])
            s_accum = np.append(
                s_accum,
                s_accum[-1] + np.linalg.norm(path[-1] - path[-2])
            )

    sx = CubicSpline(s_accum, path[:, 0], bc_type=bc)
    sy = CubicSpline(s_accum, path[:, 1], bc_type=bc)

    np.savez(f"maps/{output_name}_frenet.npz", 
             coeffs_x=sx.c, coeffs_y=sy.c, breakpoints=sx.x, total_s=s_accum.max())

    print(f"Graph-pruned path saved. Length: {s_accum.max():.2f} pixels.")
    return map_image, binary_map, skeleton_map, path, euclidean_dist



def visualize_results(map_image, binary_map, skeleton_map, path, edt_map, filename = "debug_path_img.png"):
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1. Binary Mask
    axes[0].imshow(binary_map, cmap='gray')
    axes[0].set_title("1. Binary Track Mask")
    
    # 2. EDT Map (Heatmap of distance to walls)
    edt_plot = axes[1].imshow(edt_map, cmap='jet')
    fig.colorbar(edt_plot, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("2. Euclidean Distance (EDT)")
    
    # 3. Raw Skeleton
    axes[2].imshow(map_image, cmap='gray') 
    skeleton_display = np.where(skeleton_map, 255, 0).astype(np.uint8)
    axes[2].imshow(skeleton_display, cmap='magma', alpha=0.6)
    axes[2].set_title("3. Skeleton Overlay (Ridge)")
    
    # 4. Final Path (Pruned & Ordered)
    axes[3].imshow(map_image, cmap='gray') # Underlay the original map
    if len(path) > 0:
        axes[3].plot(path[:, 0], path[:, 1], color='red', linewidth=2, label='Final Path')
        axes[3].scatter(path[0,0], path[0,1], color='green', s=100, label='Start/End', zorder=5)
    axes[3].set_title("4. Pruned Path & Spline")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Debug image saved to {filename}")

def main():
    filepath = "maps/my_mapgi.pgm"
    img, binary, skeleton, path, edt = generate_frenet(filepath)
    visualize_results(img, binary, skeleton, path, edt)

if __name__ == "__main__":
    main()    
