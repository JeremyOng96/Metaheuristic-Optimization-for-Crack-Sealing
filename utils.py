from typing import List, Dict, Set, Tuple, Optional
import heapq
import queue
import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import chain, product
from collections import OrderedDict, Counter
import math
import time
import copy


class Graph:
    def __init__(
        self, image: np.ndarray, patch_size: Tuple[int, int], verbose: Optional[bool]
    ):
        self.image = image
        self.h, self.w = self.image.shape[:2]
        self.patch_size = patch_size

        self.image_dilated = self.gridify(self.image, self.patch_size)
        self.adj_list, self.mapping, self.mapping_r = self.img2graph(
            self.image_dilated, self.patch_size, verbose
        )
        self.num_vertex = len(self.adj_list)
        self.weight_matrix = self.get_weight_matrix(self.num_vertex, self.adj_list)
        self.visited = np.asarray([False] * self.num_vertex, dtype=bool)
        self.verbose = verbose
        self.all_graphs = self.split_graph(self.adj_list)

        self.local_to_coords, self.coords_to_local = self.grid_to_coords(
            self.adj_list, self.mapping_r
        )

        print("Preprocessing of graph completed")

    def gridify(self, image, patch_size: Tuple[int, int]) -> np.ndarray:
        """
        Takes in a binary image and dilates the image to form grid images for simplification.

        Arguments
            image - binary image
            patch_size - grid size used to divide the entire image

        Return
            d_image - dilated image
        """
        patch_h, patch_w = patch_size

        img = (
            image.reshape(self.h // patch_h, patch_h, self.w // patch_w, patch_w)
            .swapaxes(1, 2)
            .reshape(-1, patch_h * patch_w)
        )
        img_dilated = np.any(img, axis=-1, keepdims=True)
        img_dilated = np.logical_or(img, img_dilated)
        img_dilated = (
            img_dilated.reshape(self.h // patch_h, self.w // patch_w, patch_h, patch_w)
            .swapaxes(1, 2)
            .reshape(self.h, self.w)
        )
        img_dilated = img_dilated[: self.h, : self.w]

        # generate a copy for graph checking using node checker
        self.img_verbose = img_dilated[..., np.newaxis]
        self.img_verbose = (
            np.repeat(img_dilated * 255, 3).reshape(self.h, self.w, 3).astype(np.uint8)
        )
        color = (0, 255, 0)
        thickness = 1
        # # draw vertical lines
        # for i in range(patch_w, self.w, patch_w):
        #     cv2.line(self.img_verbose, (i, 0), (i, self.h), color, thickness)
        # # draw horizontal lines
        # for i in range(patch_h, self.h, patch_h):
        #     cv2.line(self.img_verbose, (0, i), (self.w, i), color, thickness)

        return img_dilated.astype(int)

    def img2graph(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int],
        verbose: Optional[bool] = None,
    ):
        """
        Converts an image to graph.

        Arguments:
            image: binary image
            patch_size: transforms pixels into patches to form grids

        Return:
            m: adjacency list of image into graph
        """
        # Checks if the image is divisible by the patch_size
        h, w = image.shape
        patch_h, patch_w = patch_size
        assert (
            h % patch_h == 0 and w % patch_w == 0
        ), "image size must be divisible by global patch size"

        num_patch_r = h // patch_h
        num_patch_c = w // patch_w
        num_patches = (h * w) // (patch_h * patch_w)
        img = image

        num_cells = np.arange(0, num_patch_r * num_patch_c)
        bin_arr_r = num_cells.reshape(-1, num_patch_c)
        bin_arr_c = num_cells.reshape(-1, num_patch_c).T
        num_cells = num_cells[:, np.newaxis]

        # unroll img_gridify into row and column form
        img_r = (
            img.reshape(h // patch_h, patch_h, w // patch_w, patch_w)
            .swapaxes(1, 2)
            .reshape(-1, patch_h, patch_w)
        )

        # form adjacacency list
        adj_list = [set() for _ in range(num_patches)]
        for current_patch in range(num_patches):
            # locate the position of the current patch in grid array
            if np.all(img_r[current_patch]):
                bin_index_r, patch_index_r = np.where(bin_arr_r == current_patch)
                bin_index_c, patch_index_c = np.where(bin_arr_c == current_patch)
                neigh_index = [
                    patch_index_r[0] + 1,
                    patch_index_r[0] - 1,
                    patch_index_c[0] + 1,
                    patch_index_c[0] - 1,
                ]  # potential neighbours
                for i, n in enumerate(neigh_index):
                    if i < 2 and n >= 0 and n < num_patch_c:
                        neigh_r = bin_arr_r[bin_index_r[0], n]
                        patch_r = img_r[neigh_r, ...]
                        if np.all(patch_r):
                            adj_list[current_patch].add(neigh_r)
                            adj_list[neigh_r].add(current_patch)

                    elif i >= 2 and n >= 0 and n < num_patch_r:
                        neigh_c = bin_arr_c[bin_index_c[0], n]
                        patch_c = img_r[neigh_c, ...]
                        if np.all(patch_c):
                            adj_list[current_patch].add(neigh_c)
                            adj_list[neigh_c].add(current_patch)

        # rearrange the nodes
        node_unordered = set(chain.from_iterable(adj_list))
        node_sorted = sorted(node_unordered)
        mapping = dict()  # Maps global patch -> local patch
        adj_list_dict = OrderedDict()
        for node in node_unordered:
            mapping[node] = node_sorted.index(node)

        mapping_r = {v: k for k, v in mapping.items()}

        for i, nodes in enumerate(adj_list):
            if nodes:
                adj_list_dict[mapping[i]] = {mapping[node] for node in nodes}

        if verbose:
            self.node_checker(adj_list)

        return adj_list_dict, mapping, mapping_r

    def get_weight_matrix(self, num_vertex, adj_list) -> np.ndarray:
        weight_matrix = np.zeros((num_vertex, num_vertex))
        for i in range(num_vertex):
            for j, distance in self.propagate(i, adj_list)[0].items():
                weight_matrix[i, j] = distance

        return weight_matrix

    def split_graph(self, adj_list):
        """
        Splits the main graph into multiple graphs, this is used for multicrack purposes

        Starts splitting from node 0
        """
        adj_list = copy.deepcopy(self.adj_list)
        start_node = list(adj_list.keys())[0]
        all_graphs = []
        while True:
            adj_sublist = OrderedDict()
            dist, _ = self.propagate(start_node, adj_list)
            for node, _ in dist.items():
                adj_sublist[node] = adj_list.pop(node)

            all_graphs.append(adj_sublist)
            if len(adj_list) == 0:
                return all_graphs
            else:
                start_node = list(adj_list.keys())[0]

    def shortest_pair(self, crack_1, crack_2):
        """
        Finds a pair nodes (in global coordinates) that have the shortest distance between two cracks.
        Use Euclidean distance to find an approximation of nodes.


        Arguments:
        crack_1: Crack graph
        crack_2: Crack graph

        Return
        """
        node_1 = node_2 = None
        min_dist = float("inf")
        for i in crack_1.keys():
            for j in crack_2.keys():
                i_coords = self.local_to_coords[i]
                j_coords = self.local_to_coords[j]
                distance = np.linalg.norm(i_coords - j_coords)

                if min_dist > distance:
                    node_1 = i
                    node_2 = j

        return node_1, node_2

    def dist_pair(self, node_1, node_2):
        """
        Gets
        """
        patch_h, patch_w = self.patch_size
        img = np.zeros((self.h, self.w, 3))
        coord_node_1 = self.local_to_coords[node_1]
        coord_node_2 = self.local_to_coords[node_2]
        corner_1_h = min(coord_node_1[0], coord_node_2[0]) * self.patch_size[0]
        corner_1_w = min(coord_node_1[1], coord_node_2[1]) * self.patch_size[1]
        corner_2_h = (
            max(coord_node_1[0], coord_node_2[0]) * self.patch_size[0]
            + self.patch_size[0]
        )
        corner_2_w = (
            max(coord_node_1[1], coord_node_2[1]) * self.patch_size[1]
            + self.patch_size[1]
        )

        color = [255, 255, 255]
        start_point = (corner_1_w, corner_1_h)
        end_point = (corner_2_w, corner_2_h)

        # use existing code base to convert image to graph interpretation
        img = cv2.rectangle(img, start_point, end_point, color, thickness=-1)
        img = (img[:, :, 0] > 0).astype(int)
        img_patch = (
            img.reshape(self.h // patch_h, patch_h, self.w // patch_w, patch_w)
            .transpose(0, 2, 1, 3)
            .reshape(-1, patch_h, patch_w)
        )

        adj_list, _, mapping_r = self.img2graph(img, self.patch_size)
        local_to_coords, coords_to_local = self.grid_to_coords(adj_list, mapping_r)
        local_to_global_patch = lambda node: mapping_r[node]

        node_1_nc = coords_to_local[tuple(coord_node_1)]
        node_2_nc = coords_to_local[tuple(coord_node_2)]
        path, distance = self.a_star(node_1_nc, node_2_nc, adj_list, local_to_coords)

        path = list(map(local_to_global_patch, path))
        img_path = np.zeros_like(img_patch)
        img_path[path, ...] = True
        path_visualize = (
            img_path.reshape(self.h // patch_h, self.w // patch_w, patch_h, patch_w)
            .transpose(0, 2, 1, 3)
            .reshape(self.h, self.w)
        )
        path_visualize = path_visualize[:, :, np.newaxis].astype(int)
        path_visualize = np.repeat(path_visualize, repeats=3, axis=2) * 255

        return path, distance

    def heuristic(self, start, end, local_to_coords):
        """
        Implement the Euclidean distance heuristic for A*.
        """
        coord_1 = list(local_to_coords[start])
        coord_2 = list(local_to_coords[end])

        return math.dist(coord_1, coord_2)

    def a_star(self, start, end, adj_list, local_to_coords):
        """
        Implement the A* algorithm

        *Note*
        A* works with local 1D coordinates
        """
        visited = [False] * len(adj_list)
        final_dist = [10**9] * len(adj_list)
        potential = [10**9] * len(adj_list)
        pointer = [-1] * len(adj_list)
        pq = [(0, 0, 0, start)]
        pointer[start] = 0
        final_dist[start] = 0
        potential[start] = 0
        while pq:
            _, shortest_dist, parent, node = heapq.heappop(pq)
            visited[node] = True
            pointer[node] = parent

            # retrace path
            if node == end:
                path = [end]
                while True:
                    parent = pointer[node]
                    path.append(parent)
                    node = parent
                    if parent == start:
                        return path[::-1], shortest_dist
                    
            for vertex in adj_list[node]:
                new_dist = shortest_dist + 1
                new_potential = self.heuristic(vertex, end, local_to_coords)
                if new_dist < final_dist[vertex]:
                    final_dist[vertex] = new_dist

                if new_potential < potential[vertex]:
                    potential[vertex] = new_potential
                    heapq.heappush(pq, (new_potential, new_dist, node, vertex))

    def grid_to_coords(self, adj_list, mapping):
        local_to_coords = dict()  # transforms local patch number -> coordinate system
        coords_to_local = dict()
        for local_patch_number in adj_list:
            global_patch_number = mapping[local_patch_number]
            x = int(global_patch_number % (self.w / self.patch_size[1]))
            y = int(global_patch_number // (self.w / self.patch_size[1]))
            local_to_coords[local_patch_number] = np.asarray((y, x))  # (row,column)
            coords_to_local[(y, x)] = local_patch_number

        return local_to_coords, coords_to_local

    def node_checker(self, adj_list: Dict[int, Set[int]]):
        h, w, c = self.img_verbose.shape
        patch_h, patch_w = self.patch_size
        self.img_verbose = (
            self.img_verbose.transpose(2, 0, 1)
            .reshape(3, h // patch_h, patch_h, w // patch_w, patch_w)
            .swapaxes(2, 3)
            .reshape(3, -1, patch_h, patch_w)
            .transpose(1, 0, 2, 3)
        )
        color = [
            [[128], [128], [128]],
            [[255], [0], [0]],
            [[0], [255], [0]],
            [[0], [0], [255]],
            [[255], [255], [0]],
        ]
        color = np.tile(color, patch_h * patch_w).reshape(-1, 3, patch_h, patch_w)

        for node in range(len(adj_list)):
            # white = 0 neighbours; red = 1 neighbour; green = 2 neighbours; blue = 3 neighbours; yellow = 4 neighbours
            num_neighbours = len(adj_list[node])
            self.img_verbose[node, ...] = color[num_neighbours]

        self.img_verbose = (
            self.img_verbose.transpose(1, 0, 2, 3)
            .reshape(3, h // patch_h, w // patch_w, patch_h, patch_w)
            .transpose(0, 1, 3, 2, 4)
            .reshape(3, h, w)
            .transpose(1, 2, 0)
        )  # reverse
        self.img_verbose = self.img_verbose.astype(np.uint8).copy()
        color = (0, 255, 0)
        thickness = 1
        # draw vertical lines
        # for i in range(patch_w, self.w, patch_w):
        #     cv2.line(self.img_verbose, (i, 0), (i, self.h), color, thickness)
        # # draw horizontal lines
        # for i in range(patch_h, self.h, patch_h):
        #     cv2.line(self.img_verbose, (0, i), (self.w, i), color, thickness)

        plt.figure()
        plt.axis(False)
        plt.imshow(self.img_verbose)

    def propagate(self, start, adj_list):
        """
        Propagates a wavefront from the start node
        throughout the entire graph

        Return dist and dist_r
        dist(key = node, value = distance)
        """
        INF = 10**9
        dist = OrderedDict()
        dist_r = OrderedDict()
        dist_r = (
            dict()
        )  # this is the reverse of the distance dictionary where it maps the distance -> [nodes]
        final = [False] * self.num_vertex
        q = queue.Queue()
        q.put(start)
        dist[start] = 0
        dist_r[start] = {0}
        final[start] = True
        while not q.empty():
            node = q.get()
            for vertex in adj_list[node]:
                if not final[vertex]:
                    q.put(vertex)
                    dist[vertex] = dist[node] + 1
                    final[vertex] = True

                    if (dist[node] + 1) in dist_r.keys():
                        dist_r[dist[node] + 1].add(vertex)
                    else:
                        dist_r[dist[node] + 1] = set([vertex])

        return dist, dist_r

    def propagation_checker(self):
        """
        Checks the correctness of the propagation
        """
        h, w, c = self.img_verbose.shape
        patch_h, patch_w = self.patch_size
        img_patches = (
            self.img_verbose.transpose(2, 0, 1)
            .reshape(3, h // patch_h, patch_h, w // patch_w, patch_w)
            .swapaxes(2, 3)
            .reshape(3, -1, patch_h, patch_w)
            .transpose(1, 0, 2, 3)
        )
        for i in self.adj_list:
            color = np.array(
                [[255], [165], [0]]
            )  # orange colour for heatmap, cause i love orange
            color = np.tile(color, patch_h * patch_w).reshape(-1, 3, patch_h, patch_w)
            node = self.mapping_r[i]
            color -= self.dist[i] * 2
            img_patches[node, ...] = color

        img_verbose = (
            img_patches.transpose(1, 0, 2, 3)
            .reshape(3, h // patch_h, w // patch_w, patch_h, patch_w)
            .transpose(0, 1, 3, 2, 4)
            .reshape(3, h, w)
            .transpose(1, 2, 0)
        )  # reverse
        plt.figure()
        plt.axis(False)
        plt.imshow(img_verbose)


class Path(Graph):
    def __init__(
        self, image: np.ndarray, patch_size: Tuple[int, int], verbose: Optional[bool]
    ):
        super().__init__(image, patch_size, verbose)

    def build_path(
        self,
        path: List[int],
        adj_list: Dict[int, Set[int]],
        local_to_coords: Dict[int, Tuple[(int, int)]],
    ):
        """
        Build CCPP path from TSP path.

        Path : The path is in global 1D coordinates
        """
        i = 0
        visited = [False] * len(path)
        while True:
            node = path[i]
            next_node = path[i + 1]
            if next_node not in adj_list[node]:
                ext_path, _ = self.a_star(node, next_node, adj_list, local_to_coords)
                path = list(path[:i]) + ext_path + list(path[i + 2 :])

            i += 1
            if i == len(path) - 1:
                for index in range(len(path)):
                    visited[index] = True
                    if all(visited):
                        return path

    def one_path_checker(self, paths: List[int]):
        """
        Implement propagation checker to get metrics,
        the metrics should include
              (1) Path length
              (2) Overlap
              (3) Completeness
        """
        visited = np.asarray([False] * self.h * self.w)
        
        flatten_path = list(chain(paths))
        counter = Counter(flatten_path)
        visited[list(counter.keys())] = True
        path_len = len(flatten_path)
        num_overlap = sum([v - 1 for k, v in counter.items()])
        h, w, c = self.img_verbose.shape
        patch_h, patch_w = self.patch_size 
        img_verbose = np.zeros((self.h,self.w,3),dtype=np.uint8) + 128
    
        img_patches = (
            img_verbose.transpose(2, 0, 1)
            .reshape(3, h // patch_h, patch_h, w // patch_w, patch_w)
            .swapaxes(2, 3)
            .reshape(3, -1, patch_h, patch_w)
            .transpose(1, 0, 2, 3)
        )

        color = np.array([[0], [255], [255]])  # teal for path overlap checking
        color = np.tile(color, patch_h * patch_w).reshape(-1, 3, patch_h, patch_w)

        if self.verbose:
            for i in flatten_path:
                if visited[i]:
                    node_color = color - (counter[i] - 1)

                img_patches[i, ...] = node_color
            img_verbose = (
                img_patches.transpose(1, 0, 2, 3)
                .reshape(3, h // patch_h, w // patch_w, patch_h, patch_w)
                .transpose(0, 1, 3, 2, 4)
                .reshape(3, h, w)
                .transpose(1, 2, 0)
            )  # reverse

        img_verbose = img_verbose.astype(np.uint8).copy()

        no_fill = 0
        img_patch_true =  self.image_dilated.reshape(self.h // patch_h, patch_h, self.w // patch_w, patch_w).swapaxes(1, 2).reshape(-1, patch_h * patch_w)
        img_patch_path = (img_verbose[...,0] > 128).reshape(self.h // patch_h, patch_h, self.w // patch_w, patch_w).swapaxes(1, 2).reshape(-1, patch_h * patch_w) # 128 is used as a constant

        for i in range(img_patch_true.shape[0]):
            if i in self.mapping:
                if (np.any(img_patch_true[i]) and np.any(img_patch_path)) != True:
                    no_fill += 1
    
        return path_len, num_overlap, no_fill
    
    def check_all_path(self):
        path_len = 0
        num_overlap = 0
        no_fill = 0
        patch_h, patch_w = self.patch_size
        h,w = self.h,self.w
        img_verbose = np.zeros((self.h,self.w,3),dtype=np.uint8) + 255
    
        img_patches = (
            img_verbose.transpose(2, 0, 1)
            .reshape(3, h // patch_h, patch_h, w // patch_w, patch_w)
            .swapaxes(2, 3)
            .reshape(3, -1, patch_h, patch_w)
            .transpose(1, 0, 2, 3)
        )
        color = np.array([[255], [191], [0]])  # teal for path overlap checking
        color = np.tile(color, patch_h * patch_w).reshape(3, patch_h, patch_w)
        all_nodes = []
        all_nodes.extend(self.all_path_converted)
        all_nodes.extend(self.traverse_path)
        flatten_path = chain(*all_nodes)

        for i in flatten_path:
            img_patches[i,...] = color

        for path in self.all_path_converted:
            path_len_i, num_overlap_i, no_fill_i = self.one_path_checker(path)
            path_len += path_len_i
            num_overlap += num_overlap_i
            no_fill += no_fill_i
            for i in path:
                img_patches[i,...] = img_patches[i,...] - np.array([[[44]],[[44]],[[0]]])

        for path in self.traverse_path:
            path_len += len(path)


            
        img_verbose = (
                img_patches.transpose(1, 0, 2, 3)
                .reshape(3, h // patch_h, w // patch_w, patch_h, patch_w)
                .transpose(0, 1, 3, 2, 4)
                .reshape(3, h, w)
                .transpose(1, 2, 0)
        )  # reverse
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for path in self.all_path_converted:
            path_local = list(map(lambda x : self.mapping[x],path))
            start_coords = self.local_to_coords[path_local[0]]
            end_coords = self.local_to_coords[path_local[-1]]
            y_s, x_s = start_coords[0] * self.patch_size[0], start_coords[1] * self.patch_size[1]
            y_e, x_e = end_coords[0] * self.patch_size[0], end_coords[1] * self.patch_size[1]
            
            print(x_s,y_s,x_e,y_e)
            ax.annotate("start",(x_s,y_s))
            ax.annotate("end",(x_e,y_e))


        ax.grid(False)
        ax.set_axis_off()
        ax.invert_yaxis()
        ax.imshow(img_verbose)

        return path_len, num_overlap, no_fill
    
    def unstuck(self, start):
        """
        Generate a path to get unstuck.
        Find a shortest path to get from stuck node
        to the nearest unvisited node.
        """

        not_visited = (~self.visited).nonzero()[0]
        min_path = None
        min_dist = 10**9
        for node in not_visited:
            path, dist = self.a_star(start, node)
            if dist < min_dist:
                min_dist = dist
                min_path = path
        return min_path[1:]

    def post_path(self, paths: List[List[int]]):
        """
        Combine all paths based on the start and end nodes of each path
        """
        CONSTANT = 10000
        graph_paths = np.ones((len(paths), len(paths))) * CONSTANT
        connect_nodes = dict()
        for i in range(graph_paths.shape[0]):
            start_1 = paths[i][0]
            end_1 = paths[i][-1]
            # crack_1 = [start_1, end_1]
            for j in range(i + 1, graph_paths.shape[1]):
                start_2 = paths[j][0]
                end_2 = paths[j][-1]
                # crack_2 = [start_2, end_2]
                # all_iterations = list(product(crack_1, crack_2))
                path, dist = self.dist_pair(end_1,start_2)
                graph_paths[i,j] = dist
                graph_paths[j,i] = dist
                connect_nodes[(i,j)] = path
                # for index, k in enumerate(all_iterations):
                #     start, end = all_iterations[index]
                #     path, dist = self.dist_pair(start, end)
                #     if graph_paths[i, j] > dist:
                #         graph_paths[i, j] = dist
                #         graph_paths[j, i] = dist
                #         connect_nodes[(i, j)] = path

        graph_paths[graph_paths == CONSTANT] = 0
        
        return graph_paths, connect_nodes

    def get_statistics(self,method_name,population_num,history_dict):
        """
        Plot the statistics of methods
        """
        for k, v in history_dict.items():
            history = np.asarray(history_dict[k])
            plt.figure(figsize=(15, 10))
            ax = plt.subplot(111)
            for i in range(population_num):
                ax.plot(np.arange(self.iteration), history[:,i], label=f"population {i}")

            box = ax.get_position()
            ax.set_position(
                [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
            )  # shrink axis height by 10%
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=population_num // 2,
            )
            ax.set_title(f"{method_name} crack {k}")
            ax.set_ylabel("Distance")
            ax.set_xlabel("Iteration")
            plt.show()

    def draw_path(self):
        """
        Draws a path based on the tour
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0, self.w//self.patch_size[1]])
        ax.set_ylim([0, self.h//self.patch_size[0]])
        ax.grid(False)
        ax.set_axis_off()
        ax.invert_yaxis()
        # seal path
        for path in self.seal_path:
            local_coords = list(map(lambda x: self.local_to_coords[x], path))
            path = np.asarray(local_coords)
            x = path[:,1]
            y = path[:,0]
            colors = plt.cm.get_cmap("jet",len(local_coords))
            for i in range(len(local_coords)-1):
                ax.plot(x[i:i+2],y[i:i+2],color=colors(i),linewidth=1)
            ax.annotate("start",(x[0],y[0]))
            ax.annotate("end",(x[-1],y[-1]))

        # traverse path
        for path in self.traverse_path:
            local_coords = list(map(lambda x: (int(x // (self.w / self.patch_size[1])), int(x % (self.w/self.patch_size[1]))), path))
            path = np.asarray(local_coords)
            x = path[:,1]
            y = path[:,0]

            ax.plot(x,y,color="lightgrey")



