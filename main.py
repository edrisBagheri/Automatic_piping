import matplotlib.pyplot as plt

class PathFinder():
    def __init__(self, source,target, maze):
        self.source = source
        self.target = target
        self.path = []
        # Define the directions (up, down, left, right)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.maze = maze
        self.num_rows = len(maze)
        self.num_cols = len(maze[0])
        # Create a visited matrix to keep track of visited cells
        self.visited = [[False for _ in range(self.num_cols)] for _ in range(self.num_rows)]


    def find_path(self):
        return self.dfs(self.source[0], self.source[1])

    # Helper function to check if a given position is valid
    def is_valid(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.maze[row][col] == 0 and not self.visited[row][col]

        # Helper function to perform DFS and find all paths

    def dfs(self, row, col):
        path = []
        # Create a visited matrix to keep track of visited cells
        self.visited
        num_rows = self.num_rows
        num_cols = self.num_cols
        directions = self.directions

        path.append((row, col))
        self.visited[row][col] = True
        if (row, col) == (self.target[0], self.target[1]):
            return True

        path_is_finded = False

        while (path_is_finded == False):
            dead_end = True
            for drow, dcol in directions:
                new_row, new_col = row + drow, col + dcol

                if self.is_valid(new_row, new_col):
                    self.visited[new_row][new_col] = True
                    if (new_row, new_col) == (self.target[0], self.target[1]):
                        path_is_finded = True
                    # add the node to the path
                    path.append((new_row, new_col))
                    row, col = new_row, new_col
                    dead_end = False
                    break
            if dead_end:
                path.pop()
                row, col = path[-1][0], path[-1][1]
        # save the path in object
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


def draw_path(path):

    # Extract x and y values from coordinates
    x = [coord[1] for coord in path]
    y = [coord[0] for coord in path]

    # Plot the polyline
    plt.plot(x, y, marker='o', linestyle='-', color='blue')

    # Set plot limits
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('pipe_route')

    # Display the plot
    plt.show()


def main():
    maze = create_maze()
    test_path_finder = PathFinder((0,4), (3,6),maze)
    path = test_path_finder.find_path()
    draw_path(path)

    # print(test_path_finder.find_path())


if __name__ == '__main__':
    main()

