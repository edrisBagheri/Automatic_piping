import matplotlib.pyplot as plt
import numpy as np

class PathFinder():
    def __init__(self, sources, target, maze):
        self.sources = sources
        self.target = target
        self.path = []
        # Define the directions (up, down, left, right)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.maze = maze
        self.num_rows = len(maze)
        self.num_cols = len(maze[0])
        self.layout = []
        # Create a visited matrix to keep track of visited cells
        self.visited = [[False for _ in range(self.num_cols)] for _ in range(self.num_rows)]


    def find_path(self, start):
        return self.dfs(start[0], start[1])


    def is_valid(self, row, col):
        #check if a given position is inside the maze
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.maze[row][col] == 0

        # Helper function to perform DFS and find all paths

    def dfs(self, row, col):
        path = []
        directions = self.directions

        path.append((row, col))
        self.visited[row][col] = True
        if (row, col) == (self.target[0], self.target[1]) or (row, col) in self.layout:
            return True

        path_is_finded = False

        while (path_is_finded == False):
            dead_end = True
            for drow, dcol in directions:
                new_row, new_col = row + drow, col + dcol
                if self.is_valid(new_row, new_col):
                    if not self.visited[new_row][new_col] or (new_row, new_col) in self.layout:
                        # once we have more that one path in layout the that are visited also are valid
                        self.visited[new_row][new_col] = True
                        if (new_row, new_col) == (self.target[0], self.target[1]) or (new_row, new_col) in self.layout:
                            path_is_finded = True
                        # add the node to the path
                        path.append((new_row, new_col))
                        row, col = new_row, new_col
                        dead_end = False
                        break
            if dead_end:
                path.pop()
                row, col = path[-1][0], path[-1][1]
        # add path to layout
        self.layout.extend(path)
        return path


def create_maze():
    # for now manually creat a 2D maze
    # maze = [
    #         [0, 0, 0, 0, 0],
    #         [0, 1, 0, 1, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 1, 1, 1, 0],
    #         [0, 0, 0, 1, 0],
    #         [0, 1, 0, 0, 0]
    #     ]
    maze = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    return maze


def draw_path(paths):

    # Extract x and y values from coordinates
    for p in paths:
        x = [coord[1] for coord in p]
        y = [coord[0] for coord in p]

        # Plot the polyline
        plt.plot(x, y, marker='o', linestyle='-', color='blue')


    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('pipe_route')

    # Display the plot
    plt.show()

def sort_sources(sources, target):
    # sort targest regarding the distance in decending manner
    # we like to find the path from farest target to source
    # and reast of source branch of from that main route
    dis = np.array(sources) - np.array(target)
    sources_dis = [(np.linalg.norm(i), idx) for idx, i in enumerate(dis)]
    sources_dis = sorted(sources_dis, reverse=True)
    sorted_sources = [i[1] for i in sources_dis]
    sources_coord_dic = {i: tuple(sources[i]) for i in sorted_sources}
    coord_sources_dic = {tuple(sources[i]): i for i in sorted_sources}
    return sources_coord_dic, coord_sources_dic

def find_juction_placement(layout, coords,  target_coord_dic, coord_target_dic):
    print(layout, coords)


def finding_piping_layout(sources,target, maze):
     # this dictionaries will be sufull for general purpose application and findin the neighbouring target more efficiently
     sources_coord_dic, coord_sources_dic = sort_sources(sources,target)
     path_finder = PathFinder(sources, target, maze)
     paths = []
     for source  in  range(len(sources)):
        coords = sources_coord_dic[source]
        path = path_finder.find_path(coords)
        paths.append(path)

     draw_path(paths)
     return paths






def main():
    maze = create_maze()
    sources = [(3,6), (3,2)]
    target = (0,4)
    paths = finding_piping_layout(sources, target, maze)
    # test_path_finder = PathFinder([(3,6)], np.array([0,4]) , maze)

    # path = test_path_finder.find_path((3,6))
    # draw_path(path)

    # print(test_path_finder.find_path())


if __name__ == '__main__':
    main()


