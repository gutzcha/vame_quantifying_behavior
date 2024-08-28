import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from tqdm import tqdm


def extract_number(filename):
    # Split by '_' and take the last element, then split by '.' and take the first element
    return int(filename.split('_')[-1].split('.')[0])


def display_and_save_images_as_animation(folder_path, save_path, fps=2):
    # Get a list of image files in the folder
    # image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))])
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    # Sort file names based on the extracted number
    image_files = sorted(image_files, key=extract_number)

    # Check if the folder contains any images
    if not image_files:
        print("No images found in the folder.")
        return

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Load the first image to get dimensions
    img_path = os.path.join(folder_path, image_files[0])
    img = Image.open(img_path)
    im = ax.imshow(img)
    ax.axis('off')  # Hide the axes

    # Function to update the frame
    def update_frame(i):
        img_path = os.path.join(folder_path, image_files[i])
        img = Image.open(img_path)
        im.set_data(img)  # Update the image data
        return im,  # Return the updated artists as a tuple

    # Create a progress bar
    with tqdm(total=len(image_files), desc="Creating Animation") as pbar:
        # Function to update the progress bar
        def update_progress(*args):
            pbar.update(1)

        # Create the animation
        ani = animation.FuncAnimation(fig, update_frame, frames=len(image_files), interval=1000 // fps, blit=True)

        # Save the animation with a callback to update the progress bar
        ani.save(save_path, writer='ffmpeg', fps=fps, progress_callback=update_progress)
        print(f"Animation saved to {save_path}")

    # Optionally, display the animation
    # plt.show()


if __name__ == '__main__':
    # Example usage:
    # folder_path = r'D:\Project- Electro\VAME\materials\working_dir_2\Unsupervised Learning Tutorial with VAME-Jun30-2024\results\cropped_Rat1 probe 4-day2-free_2019-05-13-114301-0000\VAME\hmm-15\gif_frames'
    fname = 'cropped_Rat1 probe4-day1-free_2019-05-12-104236-0000'
    folder_path = r'D:\Project- Electro\VAME\materials\working_dir_2\Unsupervised Learning Tutorial with VAME-Jun30-2024\results\cropped_Rat2-probe1-sniffing-day1-Free_2019-07-14-155327-0000\VAME\hmm-15\gif_frames'

    save_path = folder_path + '.mp4'
    display_and_save_images_as_animation(folder_path, save_path, fps=30)
