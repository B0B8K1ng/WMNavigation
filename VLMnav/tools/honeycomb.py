import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import math


def draw_pointy_top_honeycomb_fixed(rows, cols, cell_size=1, output_file="fixed_pointy_honeycomb.png"):
    """
    Draw a Pointy-Top Honeycomb Grid with fixed spacing and proper alignment.

    :param rows: Number of rows.
    :param cols: Number of columns.
    :param cell_size: Radius of each hexagon.
    :param output_file: File to save the resulting honeycomb image.
    """
    # Define hexagon geometry
    hex_height = 2 * cell_size  # Height of hexagon (top to bottom)
    hex_width = math.sqrt(3) * cell_size  # Width of hexagon (side to side)
    vert_dist = 1.5 * cell_size  # Vertical distance between hexagon centers
    horiz_dist = hex_width  # Horizontal distance between hexagon centers

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))  # Adjust figure size based on grid
    ax.set_aspect('equal')

    # Generate the honeycomb grid
    for row in range(rows):
        for col in range(cols):
            # Calculate the hexagon's center
            x = col * horiz_dist
            y = row * vert_dist
            if col % 2 == 1:  # Offset odd columns downwards
                y += 0.5 * hex_height

            # Draw the hexagon
            hexagon = create_pointy_top_hexagon(x, y, cell_size)
            ax.plot(hexagon[:, 0], hexagon[:, 1], color="black")

            # Add text label for row and column
            ax.text(x, y, f"({row},{col})", ha="center", va="center", fontsize=8)

    # Remove axes for a cleaner look
    ax.axis('off')

    # Save the resulting image
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pointy-Top Honeycomb grid saved as {output_file}")


def create_pointy_top_hexagon(x_center, y_center, size):
    """
    Create a Pointy-Top hexagon centered at (x_center, y_center).

    :param x_center: X-coordinate of the hexagon center.
    :param y_center: Y-coordinate of the hexagon center.
    :param size: Radius of the hexagon.
    :return: A numpy array of shape (7, 2) containing the hexagon vertices.
    """
    angles = np.linspace(math.pi / 6, 2 * math.pi + math.pi / 6, 7)  # Start with the "top" vertex
    x_hex = x_center + size * np.cos(angles)
    y_hex = y_center + size * np.sin(angles)
    return np.column_stack((x_hex, y_hex))



# Example usage
if __name__ == "__main__":
    draw_pointy_top_honeycomb_fixed(rows=10, cols=10, cell_size=1, output_file="/file_system/vepfs/algorithm/dujun.nie/honeycomb_with_labels.png")
