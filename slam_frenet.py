import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

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

    # try:         # find longest cycle
    #     cycles = list(nx.simple_cycles(max_component))
    #     if not cycles:
    #         raise nx.NetworkXNoCycle
    #     longest_cycle = max(cycles, key = len)
    #     ordered_points = longest_cycle
    # except nx.NetworkXNoCycle:
    #     print("no cycle found")
    #     start_node = list(max.component.nodes())[0]
    #     ordered_points = list(nx.dfs_preorder_nodes(max_component, source=start_node))

    # try:         # longest cycle based on given weight
    #     basis = nx.minimum_cycle_basis(max_component, weight = 'weight')
    #     if not basis:
    #         raise nx.NetworkXNoCycle
    #     longest_basis_cycle = max(basis, key = len)
    #     sub_g = max_component.subgraph(longest_basis_cycle)
    #     start_node = longest_basis_cycle[0]
    #     ordered_points = list(nx.dfs_preorder_nodes(sub_g, source = start_node))
    # except nx.NetworkXNoCycle:
    #     print("no cycle found")
    #     start_node = list(max.component.nodes())[0]
    #     ordered_points = list(nx.dfs_preorder_nodes(max_component, source=start_node))

    # Greedy traversal
    # start_node = max(max_component.nodes(), key = lambda n: max_component.nodes[n]['depth'])
    # ordered_points = [start_node]
    # visited = {start_node}
    # while len(visited) < len(max_component.nodes()):
    #     cur = ordered_points[-1]
    #     neighbors = [n for n in max_component.neighbors(cur) if n not in visited]
    #     if not neighbors:
    #         break
    #     next_node = max(neighbors, key=lambda n: max_component.nodes[n]['depth'])
    #     ordered_points.append(next_node)
    #     visited.add(next_node)

    # maximum weighted path
    start_node = max(max_component.nodes(), key = lambda n: max_component.nodes[n]['depth'])
    for u, v in max_component.edges():
        depth_u = max_component.nodes[u]['depth']
        depth_v = max_component.nodes[v]['depth']
        avg_depth = (depth_u + depth_v) / 2.0
        max_component[u][v]['weight'] = 1.0 / (avg_depth + 1e-6)

    dists = nx.single_source_dijkstra_path_length(max_component, start_node, weight = 'weight')
    end_node = max(dists, key = dists.get)

    half1 = nx.shortest_path(max_component, start_node, end_node, weight = 'weight')

    G_temp = max_component.copy()
    G_temp.remove_nodes_from(half1[1:-1])

    try:
        half2 = nx.shortest_path(G_temp, end_node, start_node, weight = 'weight')
        ordered_points = half1 + half2[1:]
    except nx.NetworkXNoPath:
        ordered_points = half1

    if np.linalg.norm(np.array(ordered_points[0]) - np.array(ordered_points[-1])) < 20:
        if not np.array_equal(ordered_points[0], ordered_points[-1]):
            ordered_points.append(ordered_points[0])
        bc = 'periodic'
    else:
        bc = 'not-a-knot'

    ordered_points.append(ordered_points[0])
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
            s_extra = np.linalg.norm(path[-1] - path[-2])
            s_accum = np.append(s_accum, s_accum[-1] + s_extra)

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
    filepath = "maps/complex_test.pgm"
    img, binary, skeleton, path, edt = generate_frenet(filepath)
    visualize_results(img, binary, skeleton, path, edt)

if __name__ == "__main__":
    main()    
