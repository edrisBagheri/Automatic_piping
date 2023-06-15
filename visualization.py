import matplotlib.pyplot as plt

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
def draw_paths_3D(paths):

    # Extract x and y values from coordinates
    # Create a 3D plot
    color_map = {0: 'blue', 1: 'red'}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, p in enumerate(paths):
        x = [coord[0] for coord in p]
        y = [coord[1] for coord in p]
        z = [coord[2] for coord in p]


        # Plot the polyline
        ax.plot(x, y, z, 'b-', color=color_map[idx])

    # Add labels and title
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D pipe route')

    # Display the plot
    plt.show()


def main():
    # test_path  = [[(0,0), (0,1), (0,2), (1,2), (2,2)]]
    # draw_paths_2D(test_path)
    test_3D_path = [[(0,0,0), (0,0,1), (0,0,2),(0,1,2),(0,2,2),(1,2,2)]]
    draw_paths_3D(test_3D_path)
if __name__ == '__main__':
    main()