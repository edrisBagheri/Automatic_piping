import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def draw_paths_2D(paths):

    # Extract x and y values from coordinates
    for p in paths:
        y = [coord[1] for coord in p]
        x = [coord[0] for coord in p]

        # Plot the polyline
        plt.plot(x, y,  marker='o', linestyle='-', color='blue')


    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('pipe_route')

    # Display the plot
    plt.show()

class Geometry():
    def __init__(self, cabinet, boxs):
        self.boxs = boxs
        # self.n_x = cabinet.shape[0]
        # self.n_y = cabinet.shape[1]
        # self.n_z = cabinet.shape[2]
        self.cabinet = cabinet
        # self.verrt = self.get_verts
        self.edges = self.calc_edges()
    def calc_edges(self):

        # node_id = 0
        # for i in range(self.n_x):
        #     for j in range(self.n_y):
        #         for k in range(self.n_z):
        #             node_coord_map[node_id] = (i, j, k)
        #             coord_node_map[(i, j, k)] = node_id
        #             node_id += 1
        #
        # node_edge_map = {}
        # edge_node_map = {}
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
            edges.append([v1, v2])
            edges.append([v2, v3])
            edges.append([v3, v4])
            edges.append([v4, v1])

            edges.append([v1, v5])
            edges.append([v2, v6])
            edges.append([v3, v7])
            edges.append([v4, v8])

            edges.append([v5, v6])
            edges.append([v6, v7])
            edges.append([v7, v8])
            edges.append([v8, v5])

        return edges


def geometry_visualize(cabinet,boxes, target, sources):
    geom  = Geometry(cabinet,boxes)
    edges = geom.edges

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    for idx, e in enumerate(edges):
            x = [coord[0] for coord in e]
            y = [coord[1] for coord in e]
            z = [coord[2] for coord in e]
            ax.plot(x, y, z, color='black',  linewidth=0.4)
    ax.scatter(target.coord[0], target.coord[1], target.coord[2], color='red')

    for s in sources:
        ax.scatter(s.coord[0], s.coord[1], s.coord[2], color='blue')
    return ax

def draw_paths_3D(paths, ax):

    for idx, p in enumerate(paths):
        x = [coord[0] for coord in p]
        y = [coord[1] for coord in p]
        z = [coord[2] for coord in p]


        # Plot the polyline
        ax.plot(x, y, z, color='green')


    # Display the plot
    plt.show()


def main():
    # test_path  = [[(0,0), (0,1), (0,2), (1,2), (2,2)]]
    # draw_paths_2D(test_path)
    test_3D_path = [[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2), (1, 2, 2)]]
    draw_paths_3D(test_3D_path)
if __name__ == '__main__':
    main()