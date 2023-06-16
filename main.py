import numpy as np
import networkx as nx
import visualization
# from collections import deque

class Node():
    def __init__(self, direct=(1, 0, 0), coord=(0,0,0)):
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



class PathFinder():
    def __init__(self, sources, target, maze, scheme):
        self.sources = sources
        self.target = target
        # Define the directions (north(z+), south(z-), right(x+), left(x-), up(y+), down(y-))
        self.directions = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        self.maze = maze
        self.num_rows = maze.shape[0]
        self.num_cols = maze.shape[1]
        self.num_elev = maze.shape[2]
        self.t_connections = []
        self.layout = []
        # Create a visited matrix to keep track of visited cells
        self.visited = np.zeros((self.num_rows, self.num_cols, self.num_elev ), dtype=bool)
        self.enforce_connection_direction()
        self.scheme = scheme


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






    def find_path(self, start):
        if self.scheme == 0:
            return self.dfs(start[0], start[1], start[2])
        elif self.scheme == 1:
            return self.bfs(start[0], start[1], start[2])
        else:
            return self.performant_path_finding(start[0], start[1], start[2])




    def is_valid(self, row, col, elev):
        #check if a given position is inside the maze
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and 0 <= elev < self.num_elev and self.maze[row, col, elev] == 0

        # Helper function to perform DFS and find all paths


    def bfs(self, row, col, elev):

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
        path = []
        directions = self.directions
        target_coord = (self.target[0], self.target[1], self.target[2])
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



    def build_graph(self, maze):
        # Get the number of vertices in the graph
        num_row = maze.shape[0]
        num_col = maze.shape[1]
        num_elev = maze.shape[2]
        # Create an empty graph
        graph = nx.Graph()
        node_coord_map = {}
        coord_node_map = {}

        node_id = 0
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                for k in range(maze.shape[2]):
                    node_coord_map[node_id] = (i, j, k)
                    coord_node_map[(i, j, k)] = node_id
                    node_id += 1

        # define edges
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        edges = []
        node_edge_map = {}
        edge_node_map = {}

        edge_id = 0
        for node, coord in node_coord_map.items():
            if maze[coord[0], coord[1], coord[2]] == 0:
                for drow, dcol, delev in directions:
                    new_row, new_col, new_elev = coord[0] + drow, coord[1] + dcol, coord[2] + delev
                    if 0 <= new_row < num_row and 0 <= new_col < num_col and 0 <= new_elev < num_elev:
                        if maze[new_row, new_col, new_elev] == 0:
                            second_node = coord_node_map[(new_row, new_col, new_elev)]
                            edges.append((node, second_node))
                            edge_node_map[edge_id] = (node, second_node)
                            node_edge_map.setdefault(node, []).append(edge_id)
                            node_edge_map.setdefault(second_node, []).append(edge_id)
                            edge_id += 1

        for e in edges:
            graph.add_edge(e[0], e[1])

        return graph, coord_node_map, node_coord_map








def sort_sources(sources, target):
    # sort targest regarding the distance in decending manner
    # we like to find the path from closest target to source
    # since we are using BFS, the closer we are to target the faster algorithm works

    sources_coords = np.array([i.coord for i in sources])
    dis = np.array(sources_coords) - np.array(target.coord)
    sources_dis = [(np.linalg.norm(i), idx) for idx, i in enumerate(dis)]
    # sources_dis = sorted(sources_dis, reverse=True)
    sorted_sources = [i[1] for i in sources_dis]
    sources_coord_dic = {idx: tuple(sources_coords[i]) for idx, i in enumerate(sorted_sources)}
    return sources_coord_dic

def find_juction_placement(layout, coords,  target_coord_dic, coord_target_dic):
    print(layout, coords)


def finding_piping_layout(sources,target, maze, scheme):
     # this dictionaries will be sufull for general purpose application and findin the neighbouring target more efficiently
     # scheme 0: dfs, 1: bfs, 2: a_star
     path_finder = PathFinder(sources, target, maze, scheme)
     t_connections = []
     paths_coords = []
     if path_finder.scheme==2:
        graph, coord_node_map, node_coord_map = path_finder.build_graph(maze)
        sources_ids = []
        for s in sources:
            sources_ids.append(coord_node_map[s.coord])
        target_id = coord_node_map[target.coord]

        layout = [target_id]
        counter = 0
        paths = []
        for sources_id in sources_ids:
            # shortest_paths = PathFinder.find_path()
            shortest_paths = nx.multi_source_dijkstra(graph, sources=layout, target=sources_id)
            path = shortest_paths[1]
            paths.append(path)
            if counter != 0:
                t_connections.append(path[0])
            # path.pop(0) # make sure connection is not in layout twice
            layout.extend(path)
            counter += 1


        for p in paths:
            path_coord = []
            for node in p:
                path_coord.append(node_coord_map[node])
            paths_coords.append(path_coord)
     else:
         sources_coord_dic = sort_sources(sources, target)

         for source  in  range(len(sources)):
            coords = sources_coord_dic[source]
            path = path_finder.find_path(coords)
            paths_coords.append(path)
         t_connections.append(path_finder.t_connections)
         # visualization.draw_paths_3D(paths)
     return paths_coords, t_connections




def cal_material_bill(paths):
    total = 0
    for p in paths:
        total += len(p)-1
    return total

def setup_problem_1():
    # creat geometry
    maze = np.zeros((6,6,6), dtype=np.int32)
    # -----------------------------------------------------------------------------
    # geometry consist of diffrent boxes as an obstacle to be subtractet from domain
    # each box ned a boundig box
    boxes = [(0,6,0,2,0,3)]
    for box in boxes:
        maze[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = 1

    cabinet = (0,6,0,6,0,6)
    # visualization.geometry_visualize(cabinet, boxes)


    sources = [Node((0, 0, 1), (2, 0, 3)), Node((0, 0, 1), (4, 0, 3))]
    target = Node((0, 0, -1), (0, 4, 5))

    ax = visualization.geometry_visualize(cabinet, boxes, target, sources)


    return maze, sources, target, ax



def setup_problem_2():
    # create maze(adjaicency matrix) that represent the geometry
    maze = np.zeros((24, 30, 24), dtype=np.int32)
    # -----------------------------------------------------------------------------
    # geometry consist of diffrent boxes as an obstacle to be subtractet from domain
    # each box ned a boundig box
    boxes = [( 0,  8,  0, 30,  0, 12),
             ( 8, 19,  3, 27,  0, 12),
             ( 3, 19,  3,  6, 12, 24),
             ( 3, 19, 24, 27, 12, 24),
             ( 3,  6,  6, 24, 12, 20),
             ( 6, 12, 11, 19, 12, 20),
             (16, 19,  6, 24, 12, 24),
             ( 3,  6,  6, 12, 12, 24),
             ( 3,  6, 18, 24, 12, 24)
             ]
    for box in boxes:
        maze[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = 1
    cabinet = (0,24,0,30,0,24)
    # visualization.geometry_visualize(cabinet, boxes)

    sources = [Node((0, 0, 1), (9, 9, 13)), Node((0, 0, 1), (9, 23, 13))]
    target = Node((-1, 0, 0), (23, 14, 5))

    ax = visualization.geometry_visualize(cabinet, boxes, target, sources)

    return maze, sources, target, ax

def setup_problem_3():
    # create maze(adjaicency matrix) that represent the geometry
    maze = np.zeros((240, 300, 240), dtype=np.int32)
    # -----------------------------------------------------------------------------
    # geometry consist of diffrent boxes as an obstacle to be subtractet from domain
    # each box ned a boundig box
    boxes = [( 0,   80,   0, 300,  0,  120),
             ( 80, 190,  30, 270,  0,  120),
             ( 30, 190,  30,  60, 120, 240),
             ( 30, 190, 240, 270, 120, 240),
             ( 30,  60,  60, 240, 120, 200),
             ( 60, 120, 110, 190, 120, 200),
             (160, 190,  60, 240, 120, 240),
             ( 30,  60,  60, 120, 120, 240),
             ( 30,  60, 180, 240, 120, 240)
             ]
    for box in boxes:
        maze[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = 1
    cabinet = (0, 240, 0, 300, 0, 240)
    # visualization.geometry_visualize(cabinet, boxes)

    sources = [Node((0, 0, 1), (90, 90, 130)), Node((0, 0, 1), (90, 230, 130))]
    target = Node((-1, 0, 0), (230, 140, 50))

    ax = visualization.geometry_visualize(cabinet, boxes, target, sources)

    return maze, sources, target, ax




def calc_material_schedule(paths):
    pipe_layout = []
    for path in paths:
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

    return pipe_list

def main():

    maze,sources, target, ax = setup_problem_1()
    #
    # maze, sources, target, ax = setup_problem_2()

    # maze, sources, target, ax = setup_problem_3()

    paths, t_connections = finding_piping_layout(sources, target, maze, scheme=2)
    pipe_list = calc_material_schedule(paths)


    # test_path_finder = PathFinder(sources, (0, 4, 5), maze)
    # total_pipe_lenght = cal_material_bill(paths)
    # path = test_path_finder.find_path((3, 0, 3))
    # paths= []
    # paths.append(path)
    print('t connection are occure at',  t_connections)
    for p_id, p in enumerate(pipe_list):
        print("-------------------------------------")
        print("pipe: ", p_id, "lenght: ", p.lenght)
        print("segment coordinate", p.segments)
    # print('total pipe lenght:', total_pipe_lenght)
    visualization.draw_paths_3D(paths, ax)


    # print(test_path_finder.find_path())


if __name__ == '__main__':
    main()


