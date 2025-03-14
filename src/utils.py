import logging
import os
import traceback
import cv2
import numpy as np
import math
import quaternion

import  matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (200, 200, 200)

def local_to_global(position, orientation, local_point):
    """
    Transforms a local coordinate point to global coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        local_point (np.ndarray): The point in local coordinates.

    Returns:
        np.ndarray: Transformed global coordinates.
    """
    rotated_point = quaternion.rotate_vectors(orientation, local_point)
    global_point = rotated_point + position
    return global_point


def global_to_local(position, orientation, global_point):
    """
    Transforms a global coordinate point to local coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        global_point (np.ndarray): The point in global coordinates.

    Returns:
        np.ndarray: Transformed local coordinates.
    """
    translated_point = global_point - position
    inverse_orientation = np.quaternion.conj(orientation)
    local_point = quaternion.rotate_vectors(inverse_orientation, translated_point)
    return local_point


def calculate_focal_length(fov_degrees, image_width):
    """
    Calculates the focal length in pixels based on the field of view and image width.

    Args:
        fov_degrees (float): Field of view in degrees.
        image_width (int): The width of the image in pixels.

    Returns:
        float: The focal length in pixels.
    """
    fov_radians = np.deg2rad(fov_degrees)
    focal_length = (image_width / 2) / np.tan(fov_radians / 2)
    return focal_length


def local_to_image(local_point, resolution, focal_length):
    """
    Converts a local 3D point to image pixel coordinates.

    Args:
        local_point (np.ndarray): The point in local coordinates.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The pixel coordinates (x_pixel, y_pixel).
    """
    point_3d = [local_point[0], -local_point[1], -local_point[2]]  # Inconsistency between Habitat camera frame and classical convention
    if point_3d[2] == 0:
        point_3d[2] = 0.0001
    x = focal_length * point_3d[0] / point_3d[2]
    x_pixel = int(resolution[1] / 2 + x)

    y = focal_length * point_3d[1] / point_3d[2]
    y_pixel = int(resolution[0] / 2 + y)
    return x_pixel, y_pixel


def unproject_2d(x_pixel, y_pixel, depth, resolution, focal_length):
    """
    Unprojects a 2D pixel coordinate back to 3D space given depth information.

    Args:
        x_pixel (int): The x coordinate of the pixel.
        y_pixel (int): The y coordinate of the pixel.
        depth (float): The depth value at the pixel.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The 3D coordinates (x, y, z).
    """
    x = (x_pixel - resolution[1] / 2) * depth / focal_length
    y = (y_pixel - resolution[0] / 2) * depth / focal_length
    return x, -y, -depth


def agent_frame_to_image_coords(point, agent_state, sensor_state, resolution, focal_length):
    """
    Converts a point from agent frame to image coordinates.

    Args:
        point (np.ndarray): The point in agent frame coordinates.
        agent_state (6dof): The agent's state containing position and rotation.
        sensor_state (6dof): The sensor's state containing position and rotation.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple or None: The image coordinates (x_pixel, y_pixel), or None if the point is behind the camera.
    """
    global_p = local_to_global(agent_state.position, agent_state.rotation, point)
    camera_pt = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
    if camera_pt[2] > 0:
        return None
    return local_to_image(camera_pt, resolution, focal_length)


def put_text_on_image(image, text, location, font=cv2.FONT_HERSHEY_SIMPLEX, text_size=2.7, bg_color=(255, 255, 255), 
                      text_color=(0, 0, 0), text_thickness=3, highlight=True):
    """
    Puts text on an image with optional background highlighting.

    Args:
        image (np.ndarray): The image to draw on.
        text (str): The text to put on the image.
        location (str): Position for the text ('top_left', 'top_right', 'bottom_left', etc.).
        font (int): Font to use for the text.
        text_size (float): Size of the text.
        bg_color (tuple): Background color for the text (BGR).
        text_color (tuple): Color of the text (BGR).
        text_thickness (int): Thickness of the text font.
        highlight (bool): Whether to highlight the text background.

    Returns:
        np.ndarray: Image with text added.
    """
    scale_factor = image.shape[0] / 1080
    adjusted_thickness = math.ceil(scale_factor * text_thickness)
    adjusted_size = scale_factor * text_size

    assert location in ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_center', 'center'], \
        "Invalid location. Choose from 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_center', 'center'."

    img_height, img_width = image.shape[:2]
    text_size, _ = cv2.getTextSize(text, font, adjusted_size, adjusted_thickness)

    # Calculate text position
    offset = math.ceil(10 * scale_factor)
    text_x, text_y = 0, 0

    if location == 'top_left':
        text_x, text_y = offset, text_size[1] + offset
    elif location == 'top_right':
        text_x, text_y = img_width - text_size[0] - offset, text_size[1] + offset
    elif location == 'bottom_left':
        text_x, text_y = offset, img_height - offset
    elif location == 'bottom_right':
        text_x, text_y = img_width - text_size[0] - offset, img_height - offset
    elif location == 'top_center':
        text_x, text_y = (img_width - text_size[0]) // 2, text_size[1] + offset
    elif location == 'center':
        text_x, text_y = (img_width - text_size[0]) // 2, (img_height + text_size[1]) // 2

    # Draw background rectangle
    if highlight:
        cv2.rectangle(image, (text_x - offset // 2, text_y - text_size[1] - offset), 
                      (text_x + text_size[0] + offset // 2, text_y + offset), bg_color, -1)

    # Add the text
    cv2.putText(image, text, (text_x, text_y), font, adjusted_size, text_color, adjusted_thickness)
    return image

def find_intersections(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int):
    """
    Find the intersections of a line defined by two points with the image boundaries.
    Args:
        x1 (int): The x-coordinate of the first point.
        y1 (int): The y-coordinate of the first point.
        x2 (int): The x-coordinate of the second point.
        y2 (int): The y-coordinate of the second point.
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        list of tuple or None: A list of two tuples representing the intersection points 
        with the image boundaries, or None if there are not exactly two intersections.
    """
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = None  # Vertical line
        b = None

    intersections = []
    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_yh = int((img_height - b) / m)  # When y = img_height, x = (img_height - b) / m
        if 0 <= x_at_yh <= img_width:
            intersections.append((x_at_yh, img_height - 1))

    if m is not None:
        y_at_x0 = int(b)  # When x = 0, y = b
        if 0 <= y_at_x0 <= img_height:
            intersections.append((0, y_at_x0))

    if m is not None:
        y_at_xw = int(m * img_width + b)  # When x = img_width, y = m * img_width + b
        if 0 <= y_at_xw <= img_height:
            intersections.append((img_width - 1, y_at_xw))

    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_y0 = int(-b / m)  # When y = 0, x = -b / m
        if 0 <= x_at_y0 <= img_width:
            intersections.append((x_at_y0, 0))

    if m is None:
        intersections.append((x1, img_height - 1))  # Bottom edge
        intersections.append((x1, 0))  # Top edge

    if len(intersections) == 2:
        return intersections
    return None

def depth_to_height(depth_image, hfov, camera_position, camera_orientation):
    """
    Converts depth image to a height map using camera parameters.

    Args:
        depth_image (np.ndarray): The input depth image.
        hfov (float): Horizontal field of view in degrees.
        camera_position (np.ndarray): The global position of the camera.
        camera_orientation (quaternion.quaternion): The camera's quaternion orientation.

    Returns:
        np.ndarray: Global height map derived from depth image.
    """
    img_height, img_width = depth_image.shape
    focal_length_px = img_width / (2 * np.tan(np.radians(hfov / 2)))

    i_idx, j_idx = np.indices((img_height, img_width))
    x_prime = (j_idx - img_width / 2)
    y_prime = (i_idx - img_height / 2)

    x_local = x_prime * depth_image / focal_length_px
    y_local = y_prime * depth_image / focal_length_px
    z_local = depth_image

    local_points = np.stack((x_local, -y_local, -z_local), axis=-1)
    global_points = local_to_global(camera_position, camera_orientation, local_points)

    return global_points[:, :, 1]  # Return height map

def log_exception(e):
    """Logs an exception with traceback information."""
    tb = traceback.extract_tb(e.__traceback__)
    for frame in tb:
        logging.error(f"Exception in {frame.filename} at line {frame.lineno}")
    logging.error(f"Error: {e}")


def create_gif(image_dir, h, w, interval=600, agent_cls='ObjectNavAgent'):
    """
    Creates a GIF animation from images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        interval (int): Interval between frames in milliseconds.

    Returns:
        None: Saves the GIF animation in the directory.
    """
    # Create a figure that tightly matches the size of the images (1920x1080)
    #fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    #fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    if agent_cls == 'ObjectNavAgent':
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
        ax.set_position([0, 0, 1, 1])  # Remove all padding
        ax.axis('off')

        frames = []
        step_num = len([name for name in os.listdir(image_dir) if 'step' in name])

        # Process up to 80 steps
        for i in range(min(step_num, 80)):
            try:
                img = cv2.imread(f"{image_dir}/step{i}/color_sensor.png")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = [ax.imshow(img_rgb, animated=True)]
                frames.append(frame)

                img_copy = cv2.imread(f"{image_dir}/step{i}/color_sensor_chosen.png")
                img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                frame_copy = [ax.imshow(img_copy_rgb, animated=True)]
                frames.append(frame_copy)

            except Exception as e:
                continue

        # Add a black frame at the end
        #black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        #black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        frame_black = [ax.imshow(black_frame_rgb, animated=True)]
        frames.append(frame_black)

        # Create the animation
        ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

        # Save the animation
        ani.save(f'{image_dir}/animation.gif', writer='imagemagick')
        logging.info('GIF animation saved successfully!')

        # Clear the frames list after saving the animation
        frames.clear()

        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        ax.set_position([0, 0, 1, 1])  # Remove all padding
        ax.axis('off')

        frames = []
        step_num = len([name for name in os.listdir(image_dir) if 'step' in name])

        # Process up to 80 steps
        for i in range(min(step_num, 80)):
            try:
                img_origin = cv2.imread(f"{image_dir}/step{i}/color_origin.png")
                img_origin_rgb = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
                frame_origin = [ax.imshow(img_origin_rgb, animated=True)]
                frames.append(frame_origin)

                img = cv2.imread(f"{image_dir}/step{i}/color_sensor.png")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = [ax.imshow(img_rgb, animated=True)]
                frames.append(frame)


                img_copy = cv2.imread(f"{image_dir}/step{i}/color_sensor_chosen.png")
                img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                frame_copy = [ax.imshow(img_copy_rgb, animated=True)]
                frames.append(frame_copy)

            except Exception as e:
                continue

        # Add a black frame at the end
        # black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        frame_black = [ax.imshow(black_frame_rgb, animated=True)]
        frames.append(frame_black)

        # Create the animation
        ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

        # Save the animation
        ani.save(f'{image_dir}/animation.gif', writer='imagemagick')
        logging.info('GIF animation saved successfully!')

        # Clear the frames list after saving the animation
        frames.clear()
        plt.close(fig)

    # Explicitly delete objects to free memory
    del frames, ani, fig, ax
    import gc
    gc.collect()  # Trigger garbage collection to ensure memory is released


def create_gif_voxel(image_dir, h, w, interval=600):
    """
    Creates a GIF animation from images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        interval (int): Interval between frames in milliseconds.

    Returns:
        None: Saves the GIF animation in the directory.
    """
    # Create a figure that tightly matches the size of the images (1920x1080)
    #fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    #fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.set_position([0, 0, 1, 1])  # Remove all padding
    ax.axis('off')

    frames = []
    step_num = len([name for name in os.listdir(image_dir) if 'step' in name])
    # Process up to 80 steps
    for i in range(min(step_num, 80)):
        try:
            img = cv2.imread(f"{image_dir}/step{i}/voxel_map.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

            img_copy = cv2.imread(f"{image_dir}/step{i}/voxel_map_chosen.png")
            img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            frame_copy = [ax.imshow(img_copy_rgb, animated=True)]
            frames.append(frame_copy)

        except Exception as e:
            continue

    # Add a black frame at the end
    #black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    #black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
    frame_black = [ax.imshow(black_frame_rgb, animated=True)]
    frames.append(frame_black)

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation
    ani.save(f'{image_dir}/animation_voxel_map.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

    # Clear the frames list after saving the animation
    frames.clear()

    # Explicitly delete objects to free memory
    plt.close(fig)

    del frames, ani, fig, ax

    import gc
    gc.collect()  # Trigger garbage collection to ensure memory is released

def create_gif_nav(image_dir, h, w, interval=600):
    """
    Creates a GIF animation from images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        interval (int): Interval between frames in milliseconds.

    Returns:
        None: Saves the GIF animation in the directory.
    """
    # Create a figure that tightly matches the size of the images (1920x1080)
    #fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    #fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.set_position([0, 0, 1, 1])  # Remove all padding
    ax.axis('off')

    frames = []
    step_num = len([name for name in os.listdir(image_dir) if 'step' in name])
    # Process up to 80 steps
    for i in range(min(step_num, 80)):
        try:
            img = cv2.imread(f"{image_dir}/step{i}/nav_map.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

        except Exception as e:
            continue

    # Add a black frame at the end
    #black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    #black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
    frame_black = [ax.imshow(black_frame_rgb, animated=True)]
    frames.append(frame_black)

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation
    ani.save(f'{image_dir}/animation_nav_map.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

    # Clear the frames list after saving the animation
    frames.clear()

    # Explicitly delete objects to free memory
    plt.close(fig)

    del frames, ani, fig, ax

    import gc
    gc.collect()  # Trigger garbage collection to ensure memory is released

def create_gif_cvalue(image_dir, h, w, interval=600):
    """
    Creates a GIF animation from images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        interval (int): Interval between frames in milliseconds.

    Returns:
        None: Saves the GIF animation in the directory.
    """
    # Create a figure that tightly matches the size of the images (1920x1080)
    #fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    #fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.set_position([0, 0, 1, 1])  # Remove all padding
    ax.axis('off')

    frames = []
    step_num = len([name for name in os.listdir(image_dir) if 'step' in name])
    # Process up to 80 steps
    for i in range(min(step_num, 80)):
        try:
            img = cv2.imread(f"{image_dir}/step{i}/cvalue_map.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

        except Exception as e:
            continue

    # Add a black frame at the end
    #black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    #black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
    frame_black = [ax.imshow(black_frame_rgb, animated=True)]
    frames.append(frame_black)

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation
    ani.save(f'{image_dir}/animation_cvalue_map.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

    # Clear the frames list after saving the animation
    frames.clear()

    # Explicitly delete objects to free memory
    plt.close(fig)

    del frames, ani, fig, ax

    import gc
    gc.collect()  # Trigger garbage collection to ensure memory is released


