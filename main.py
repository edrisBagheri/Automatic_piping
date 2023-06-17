import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from collections import deque

class Node():
    def __init__(self, direct=(1, 0, 0), coord=(0, 0, 0)):
        self.direct = direct
        self.coord = coord


class Pipe():
    def __init__(self, segments):
        self.segments = segments
        self.lenght = self.calc_lengh()

    def calc_lengh(self):
        start = self.segments[0]
        end = self.segments[-1]
        seg = np.array([start, end])
        len = np.diff(seg, axis=0)
        return np.linalg.norm(len).astype(np.int32)


class Geometry():
    def __init__(self, cabinet, boxes):
        self.boxs = boxes
        self.cabinet = cabinet
        self.edges = self.calc_edges()
    def calc_edges(self):

        all_shapes = []
        all_shapes.append(self.cabinet)
        all_shapes.extend(self.boxs)
        edges = []
        for x_min, x_max, y_min, y_max, z_min, z_max in all_shapes:
            v1 = [x_min, y_min, z_min]
            v2 = [x_max, y_min, z_min]
            v3 = [x_max, y_max, z_min]
            v4 = [x_min, y_max, z_min]
            v5 = [x_min, y_min, z_max]
            v6 = [x_max, y_min, z_max]
            v7 = [x_max, y_max, z_max]
            v8 = [x_min, y_max, z_max]
            edges.extend([[v1, v2], [v2, v3], [v3, v4], [v4, v1]])
            edges.extend([[v1, v5], [v2, v6], [v3, v7], [v4, v8]])
            edges.extend([[v5, v6], [v6, v7], [v7, v8], [v8, v5]])


        return edges


class PipeLayoutAnalyzer():
    def __init__(self, sources, target, cabinet, boxes, algorithm=2):
        self.sources = sources
        self.target = target
        self.cabinet = cabinet
        self.boxes = boxes
        # Define the directions (north(z+), south(z-), right(x+), left(x-), up(y+), down(y-))
        self.directions = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        self.build_maze_matrix()
        self.num_rows = self.maze.shape[0]
        self.num_cols = self.maze.shape[1]
        self.num_elev = self.maze.shape[2]
        self.enforce_connection_direction()
        self.build_graph()
        self.t_connections = []
        self.layout = []
        # Create a visited matrix to keep track of visited cells
        self.visited = np.zeros((self.num_rows, self.num_cols, self.num_elev ), dtype=bool)

        self.algorithm = algorithm

    def build_maze_matrix(self):
        # create maze(adjaicency matrix) that represent the geometry
        self.maze = np.zeros((self.cabinet[1], self.cabinet[3], self.cabinet[5]), dtype=np.int32)
        # -----------------------------------------------------------------------------
        # geometry consist of diffrent boxes as an obstacle to be subtractet from domain
        # each box ned a boundig box
        for box in self.boxes:
            self.maze[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = 1



    def enforce_connection_direction(self):
        # consider a obstacle in maze all directions except node (inlet/outlet) connection
        connection_nodes = []
        connection_nodes.extend(self.sources)
        connection_nodes.append(self.target)
        for s in connection_nodes:
            coord = s.coord
            for drow, dcol, delev in self.directions:
                new_row, new_col, new_elev = coord[0] + drow, coord[1] + dcol, coord[2] + delev
                if (new_row, new_col, new_elev) != (coord[0] + s.direct[0], coord[1] + s.direct[1], coord[2] + s.direct[2]):
                    if self.is_valid(new_row, new_col, new_elev):
                        self.maze[new_row, new_col, new_elev] = 1

    def find_path(self):
        paths_coords = []
        sources_ids = []
        for s in self.sources:
            sources_ids.append(self.coord_node_map[s.coord])
        target_id = self.coord_node_map[self.target.coord]

        layout = [target_id]
        counter = 0
        paths = []

        for sources_id in sources_ids:
            # shortest_paths = PathFinder.find_path()
            path = []
            if self.algorithm == 2:
                shortest_paths = nx.multi_source_dijkstra(self.graph, sources=layout, target=sources_id)
                path.extend(shortest_paths[1])
            elif self.algorithm == 1:
                # use bfs
                coords =  self.node_coord_map[sources_id]
                path_coords = self.bfs(coords[0], coords[1], coords[2])
                path.extend([self.coord_node_map[coords]for coords in path_coords])
            else:
                # use dfs
                coords =  self.node_coord_map[sources_id]
                path_coords = self.dfs(coords[0], coords[1], coords[2])
                path.extend([self.coord_node_map[coords]for coords in path_coords])

            # path = shortest_paths[1]
            paths.append(path)
            if counter != 0:
                self.t_connections.append(path[0])
            # path.pop(0) # make sure connection is not in layout twice
            layout.extend(path)
            counter += 1

        for p in paths:
            path_coord = []
            for node in p:
                path_coord.append(self.node_coord_map[node])
            paths_coords.append(path_coord)
        self.paths_coords = paths_coords


    def is_valid(self, row, col, elev):
        #check if a given position is inside the maze
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and 0 <= elev < self.num_elev and self.maze[row, col, elev] == 0



    def bfs(self, row, col, elev):
        """
        Perform Breadth-First Search (BFS) to find a path from the given start coordinates to the target.
        """
        target_coord = self.target.coord
        directions = self.directions
        start = ( row, col, elev)
        # queue = deque([(start, [])])
        queue = [(start, [])]
        visited = set([start])
        self.visited[row, col, elev] = True

        while queue:
            # current, path = queue.popleft()
            current, path = queue[0][0], queue[0][1]
            queue.pop(0)
            row, col, elev = current

            if current == target_coord or current in self.layout:

                if self.layout:
                    path += [current]
                    self.layout.extend(path)
                else:
                    path += [current]
                    self.layout.extend(path)
                    self.t_connections.append(current)
                return path

            for drow, dcol, delev in directions:
                # nx, ny = x + dx, y + dy
                new_row, new_col, new_elev = row + drow, col + dcol, elev + delev
                if self.is_valid(new_row, new_col, new_elev):
                    if not self.visited[new_row, new_col, new_elev] or (new_row, new_col, new_elev) in self.layout:
                        queue.append(((new_row, new_col, new_elev), path + [current]))
                        visited.add((new_row, new_col, new_elev))

    def dfs(self, row, col, elev):
        """
        Perform Depth-First Search (DFS) to find a path from the given start coordinates to the target.
        """
        path = []
        directions = self.directions
        target_coord = (self.target.coord[0], self.target.coord[1], self.target.coord[2])
        path.append((row, col, elev))
        self.visited[row, col, elev] = True
        if (row, col, elev) == target_coord or (row, col, elev) in self.layout:
            return True

        path_is_finded = False

        while (path_is_finded == False):
            dead_end = True
            for drow, dcol, delev in directions:
                new_row, new_col, new_elev = row + drow, col + dcol, elev + delev
                if self.is_valid(new_row, new_col, new_elev):
                    if not self.visited[new_row, new_col, new_elev] or (new_row, new_col, new_elev) in self.layout:
                        # once we have more that one path in layout the that are visited also are valid
                        self.visited[new_row, new_col, new_elev] = True
                        new_coord = (new_row, new_col, new_elev)

                        if new_coord == target_coord or (new_row, new_col, new_elev) in self.layout:
                            path_is_finded = True
                        # add the node to the path
                        path.append((new_row, new_col, new_elev))
                        row, col, elev = new_row, new_col, new_elev
                        dead_end = False
                        break
            if dead_end:
                # backtracking
                path.pop()
                row, col, elev = path[-1][0], path[-1][1], path[-1][2]
        # add path to layout
        self.layout.extend(path)
        return path


    def build_graph(self):
        """
          Build a graph representation of the maze using NetworkX.
        """
        graph = nx.Graph()
        node_coord_map = {} # Map of node IDs to their coordinates
        coord_node_map = {} # Map of coordinates to their node IDs

        node_id = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                for k in range(self.num_elev):
                    node_coord_map[node_id] = (i, j, k)
                    coord_node_map[(i, j, k)] = node_id
                    node_id += 1

        # define edges
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        edges = []
        for node, coord in node_coord_map.items():
            if self.maze[coord[0], coord[1], coord[2]] == 0:
                for drow, dcol, delev in directions:
                    new_row, new_col, new_elev = coord[0] + drow, coord[1] + dcol, coord[2] + delev
                    if 0 <= new_row < self.num_rows and 0 <= new_col < self.num_cols and 0 <= new_elev < self.num_elev:
                        if self.maze[new_row, new_col, new_elev] == 0:
                            second_node = coord_node_map[(new_row, new_col, new_elev)]
                            edges.append((node, second_node))


        for e in edges:
            graph.add_edge(e[0], e[1])

        self.graph = graph
        self.coord_node_map = coord_node_map
        self.node_coord_map = node_coord_map

    def display_geometry(self):
        geom = Geometry(self.cabinet, self.boxes)
        edges = geom.edges

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        for idx, e in enumerate(edges):
            x = [coord[0] for coord in e]
            y = [coord[1] for coord in e]
            z = [coord[2] for coord in e]
            ax.plot(x, y, z, color='gray', linewidth=0.3)
        ax.scatter(self.target.coord[0], self.target.coord[1], self.target.coord[2], color='red')

        for s in self.sources:
            ax.scatter(s.coord[0], s.coord[1], s.coord[2], color='blue')
        self.ax =ax

    def display_piping(self):
        for idx, p in enumerate(self.paths_coords):
            x = [coord[0] for coord in p]
            y = [coord[1] for coord in p]
            z = [coord[2] for coord in p]

            # Plot the polyline
            self.ax.plot(x, y, z, color='green')

        # Display the plot
        plt.show()

    def calc_material_schedule(self):
        pipe_layout = []
        for path in self.paths_coords:
            pipe_layout.extend(path)

        diff = np.diff(np.array(pipe_layout), axis=0)

        pipe = [pipe_layout[0]]
        temp_direction = diff[0]

        pipes = []
        for seg_id, segment in enumerate(pipe_layout[1:]):
            if np.all(temp_direction == diff[seg_id]):
                pipe.append(segment)
            else:
                pipes.append(pipe)
                pipe = []
                temp_direction = diff[seg_id]

        pipe_list = []
        for p in pipes:
            if p:
                pipe_list.append(Pipe(p))

        self.pipe_list = pipe_list


    def write_schedule(self):
        with open('pipe_layout_results.txt', 'w') as f:
            f.write('Pipe Layout schedule\n\n')

            f.write('T Connections\n')
            f.write('T connections occur at\n')

            for i in range(len(self.t_connections)):
                f.write(str(self.node_coord_map[self.t_connections[i]]))
            f.write("\n")
            f.write("---------------------\n")
            f.write('Pipes\n')
            for pipe_id, pipe in enumerate(self.pipe_list):
                f.write("pipe_id: " + str(pipe_id) + "pipe lenght: " + str(pipe.lenght) + "\n")


def setup_test(test):
    test_cases = {
        1: {
            'cabinet': (0, 24, 0, 30, 0, 24),
            'boxes': [
                (0, 8, 0, 30, 0, 12),
                (8, 19, 3, 27, 0, 12),
                (3, 19, 3, 6, 12, 24),
                (3, 19, 24, 27, 12, 24),
                (3, 6, 6, 24, 12, 20),
                (6, 12, 11, 19, 12, 20),
                (16, 19, 6, 24, 12, 24),
                (3, 6, 6, 12, 12, 24),
                (3, 6, 18, 24, 12, 24)
            ],
            'sources': [Node((0, 0, 1), (9, 9, 13)), Node((0, 0, 1), (9, 23, 13))],
            'target': Node((-1, 0, 0), (23, 14, 5)),
            'algorithm': 2
        },
        2: {
            'cabinet': (0, 6, 0, 6, 0, 6),
            'boxes': [(0, 6, 0, 2, 0, 3)],
            'sources': [Node((0, 0, 1), (2, 0, 3)), Node((0, 0, 1), (4, 0, 3))],
            'target': Node((0, 0, -1), (0, 4, 5)),
            'algorithm': 1
        },
        3: {
            'cabinet': (0, 6, 0, 6, 0, 6),
            'boxes': [(0, 6, 0, 2, 0, 3)],
            'sources': [Node((0, 0, 1), (2, 0, 3)), Node((0, 0, 1), (4, 0, 3))],
            'target': Node((0, 0, -1), (0, 4, 5)),
            'algorithm': 0
        }
    }

    if test in test_cases:
        case = test_cases[test]
        return case['sources'], case['target'], case['cabinet'], case['boxes'], case['algorithm']
    else:
        raise ValueError("Invalid test case. Please select a valid test case.")


def main():
    """
        Main function for executing the pipe layout analysis.

        Please select:
        - test 1 for analyzing the geometry of the project,
        - test 2 for a small test using the BFS algorithm,
        - test 3 for a small test using the DFS algorithm.

        The 'algorithm' variable should be set to:
        - 2 for Dijkstra's algorithm,
        - 1 for BFS (Breadth-First Search),
        - 0 for DFS (Depth-First Search).
        """

    sources, target, cabinet, boxes, algorithm = setup_test(test=1)
    analyzer  = PipeLayoutAnalyzer(sources, target, cabinet, boxes, algorithm)
    analyzer .find_path()
    analyzer .calc_material_schedule()
    analyzer .write_schedule()
    analyzer .display_geometry()
    analyzer .display_piping()


if __name__ == '__main__':
    main()


