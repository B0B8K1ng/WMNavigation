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

def create_gif(image_dir, out_dir, h, w, interval=600):
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
    ani.save(f'{out_dir}/animation.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

    # Clear the frames list after saving the animation
    frames.clear()
    plt.close(fig)

    # Explicitly delete objects to free memory
    del frames, ani, fig, ax
    import gc
    gc.collect()  # Trigger garbage collection to ensure memory is released

def create_gif_nav(image_dir, out_dir, h, w, interval=600):
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

            img = cv2.imread(f"{image_dir}/step{i}/nav_map.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

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
    ani.save(f'{out_dir}/animation_nav_map.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

    # Clear the frames list after saving the animation
    frames.clear()

    # Explicitly delete objects to free memory
    plt.close(fig)

    del frames, ani, fig, ax

    import gc
    gc.collect()  # Trigger garbage collection to ensure memory is released

def create_gif_evalue(image_dir, out_dir, h, w, interval=600):
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
            img = cv2.imread(f"{image_dir}/step{i}/evalue_map.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

            img = cv2.imread(f"{image_dir}/step{i}/evalue_map.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

            img = cv2.imread(f"{image_dir}/step{i}/evalue_map.png")
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
    ani.save(f'{out_dir}/animation_evalue_map.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

    # Clear the frames list after saving the animation
    frames.clear()

    # Explicitly delete objects to free memory
    plt.close(fig)

    del frames, ani, fig, ax

    import gc
    gc.collect()  # Trigger garbage collection to ensure memory is released

import re
def extract_subtask(file_path):
    # 打开并读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式匹配subtask后面的内容
    match = re.search(r"'subtask':\s*'([^']*)'", content)

    if match:
        # 提取匹配到的内容
        subtask_content = match.group(1)
        print("Extracted subtask content:", subtask_content)
    else:
        print("No subtask found in the file.")
    return subtask_content + '.'

def create_gif_text(image_dir, out_dir, h, w, interval=600, font_size=50, padding=20):
    """
    Creates a GIF animation from images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        out_dir (str): Output directory for the GIF.
        h (int): Height of the image in pixels.
        w (int): Width of the image in pixels.
        interval (int): Interval between frames in milliseconds.
        font_size (int): Font size for the text.
        padding (int): Vertical padding between the two text lines.

    Returns:
        None: Saves the GIF animation in the directory.
    """
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.set_position([0, 0, 1, 1])  # Remove all padding
    ax.axis('off')

    frames = []

    # Use a built-in OpenCV font (e.g., FONT_HERSHEY_SIMPLEX)
    font = cv2.FONT_HERSHEY_SIMPLEX

    step_num = len([name for name in os.listdir(image_dir) if 'step' in name])

    # Process up to 80 steps
    for i in range(min(step_num, 80)):
        try:
            # Create a white background image
            img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

            subtask = extract_subtask(f"{image_dir}/step{i}/details.txt")
            last_subtask = extract_subtask(f"{image_dir}/step{i-1}/details.txt") if i > 0 else None

            # Define the text
            text1 = f"Subtask: {subtask}"
            text2 = f"Last Subtask: {last_subtask}"

            # Get the size of the text to calculate positioning
            (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_size / 30, 2)
            (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_size / 30, 2)

            # Calculate positions to center the text vertically and left-align
            position1 = (10, (h - text1_height - text2_height - padding) // 2 + text1_height)
            position2 = (10, position1[1] + text1_height + padding)

            # Put the text on the image
            cv2.putText(img, text1, position1, font, font_size / 30, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, text2, position2, font, font_size / 30, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imwrite("/file_system/vepfs/algorithm/dujun.nie/1.png", img)

            img = cv2.imread("/file_system/vepfs/algorithm/dujun.nie/1.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)
            frames.append(frame)
            frames.append(frame)

        except Exception as e:
            logging.error(f"Error processing step {i}: {e}")
            continue

    # Add a black frame at the end
    black_frame = 255 * np.ones((h, w, 3), dtype=np.uint8)
    black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
    frame_black = plt.imshow(black_frame_rgb, animated=True)
    frames.append([frame_black])

    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation as GIF
    ani.save(f'{out_dir}/animation_text.gif', writer='imagemagick')

    logging.info('GIF animation saved successfully!')

    # Clear the frames list after saving the animation
    frames.clear()

    # Close the figure and free memory
    plt.close(fig)

dir = '/file_system/vepfs/algorithm/dujun.nie/code/WMNav/logs/ObjectNav_baseline_v8_flash_2_mp3d_another/30_of_50/1328_Z6MFQCViBuw/' #552_880 /file_system/vepfs/algorithm/dujun.nie/code/WMNav/logs/ObjectNav_baseline_v8_flash_2_mp3d_another/0_of_50/0_2azQ1b91cZZ/
# /file_system/vepfs/algorithm/dujun.nie/code/WMNav/logs/ObjectNav_baseline_v8_flash_2_mp3d_another/10_of_50/454_EU6Fwq7SyZv/ /file_system/vepfs/algorithm/dujun.nie/code/WMNav/logs/ObjectNav_baseline_v8_flash_2_mp3d_another/30_of_50/1336_Z6MFQCViBuw/
out_dir = '/file_system/vepfs/algorithm/dujun.nie/'
create_gif(dir, out_dir, 480, 640)
create_gif_nav(dir, out_dir, 1800, 1800)
create_gif_evalue(dir, out_dir, 1800, 1800)
create_gif_text(dir, out_dir, 200, 1500)