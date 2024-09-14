import torch
import cv2
import numpy as np
from tkinter import Tk, filedialog, Scale, HORIZONTAL, IntVar, Canvas, Scrollbar, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
from torchvision.transforms import Compose
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt

# Load the MiDaS model
model_type = "DPT_Large"  # MiDaS model type
model = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

def select_file():
    # Open file dialog
    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    return file_path

def crop_to_content(image):
    # Convert to grayscale to find contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to get the bounding box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image

def generate_depth_map(img, progress):
    progress.set(0)
    root.update_idletasks()

    # Apply the transform to the image
    input_batch = transform(img).to(device)

    # Progress update
    progress.set(30)
    root.update_idletasks()

    # Predict the depth
    with torch.no_grad():
        prediction = model(input_batch)

    # Resize the output to match the cropped image size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Progress update
    progress.set(100)
    root.update_idletasks()
    
    depth_map = prediction.cpu().numpy()
    return depth_map

def create_layers(image, depth_map, threshold=0.5):
    # Normalize depth map to 0-1
    depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Create a mask for the foreground (closer to the camera)
    foreground_mask = depth_map_norm > threshold

    # Convert mask to uint8 for inpainting
    mask = np.uint8(foreground_mask * 255)

    # Separate foreground and background using the mask
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return foreground, background

def create_animation(foreground, background, image_size, fps, duration, speed_multiplier, direction, progress):
    fig, ax = plt.subplots(figsize=(image_size[1] / 100, image_size[0] / 100), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Precompute frames
    frames = []
    motion_amplitude = 10  # Base amplitude for movement
    motion_speed_factor = speed_multiplier * (4 / duration)  # Speed adjustment based on duration and multiplier

    for t in np.linspace(0, duration, int(fps * duration)):
        ax.clear()
        ax.axis('off')

        # Determine motion based on selected direction
        if direction == "vertical":
            shift_x = 0
            shift_y = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
        elif direction == "horizontal":
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
            shift_y = 0
        elif direction == "circle_clockwise":
            shift_x = int(motion_amplitude * np.cos(2 * np.pi * t * motion_speed_factor))
            shift_y = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
        elif direction == "circle_counterclockwise":
            shift_x = int(motion_amplitude * np.cos(-2 * np.pi * t * motion_speed_factor))
            shift_y = int(motion_amplitude * np.sin(-2 * np.pi * t * motion_speed_factor))
        elif direction == "zoom_in":
            scale_factor = 1 + 0.05 * np.sin(2 * np.pi * t * motion_speed_factor)
            foreground = cv2.resize(foreground, None, fx=scale_factor, fy=scale_factor)
            background = cv2.resize(background, None, fx=scale_factor, fy=scale_factor)
            shift_x = 0
            shift_y = 0
        elif direction == "zoom_out":
            scale_factor = 1 - 0.05 * np.sin(2 * np.pi * t * motion_speed_factor)
            foreground = cv2.resize(foreground, None, fx=scale_factor, fy=scale_factor)
            background = cv2.resize(background, None, fx=scale_factor, fy=scale_factor)
            shift_x = 0
            shift_y = 0
        elif direction == "zoom_from_center":
            scale_factor = 1 + 0.05 * np.cos(2 * np.pi * t * motion_speed_factor)
            foreground = cv2.resize(foreground, None, fx=scale_factor, fy=scale_factor)
            background = cv2.resize(background, None, fx=scale_factor, fy=scale_factor)
            shift_x = 0
            shift_y = 0
        elif direction == "perspective":
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
            shift_y = shift_x
        elif direction == "diagonal":
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
            shift_y = shift_x
        elif direction == "zoom_from_corner":
            scale_factor = 1 + 0.1 * np.sin(2 * np.pi * t * motion_speed_factor)
            foreground = cv2.resize(foreground, None, fx=scale_factor, fy=scale_factor)
            background = cv2.resize(background, None, fx=scale_factor, fy=scale_factor)
            shift_x = -shift_x / 2
            shift_y = -shift_y / 2
        elif direction == "rotation":
            angle = 10 * np.sin(2 * np.pi * t * motion_speed_factor)
            M_rotate = cv2.getRotationMatrix2D((foreground.shape[1]//2, foreground.shape[0]//2), angle, 1)
            shifted_foreground = cv2.warpAffine(foreground, M_rotate, (foreground.shape[1], foreground.shape[0]))
            shifted_background = cv2.warpAffine(background, M_rotate, (background.shape[1], background.shape[0]))
            shift_x = 0
            shift_y = 0
        elif direction == "sway":
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
            shift_y = int(motion_amplitude / 2 * np.cos(2 * np.pi * t * motion_speed_factor))
        elif direction == "spiral":
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor) * np.cos(2 * np.pi * t * motion_speed_factor))
            shift_y = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor) * np.sin(2 * np.pi * t * motion_speed_factor))
        elif direction == "bounce":
            shift_y = int(abs(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor)))
            shift_x = 0
        elif direction == "tilt":
            shift_x = int(motion_amplitude * np.tan(2 * np.pi * t * motion_speed_factor))
            shift_y = 0
        else:  # default to horizontal
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
            shift_y = 0

        # Apply shifting if not using rotation or zoom transformations
        if direction not in ["rotation", "zoom_in", "zoom_out", "zoom_from_center", "zoom_from_corner"]:
            # Shift foreground and background separately
            M_foreground = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted_foreground = cv2.warpAffine(foreground, M_foreground, (foreground.shape[1], foreground.shape[0]))

            M_background = np.float32([[1, 0, -shift_x / 2], [0, 1, -shift_y / 2]])
            shifted_background = cv2.warpAffine(background, M_background, (background.shape[1], background.shape[0]))

        # Ensure foreground remains opaque
        combined = shifted_background.copy()
        foreground_indices = shifted_foreground > 0
        combined[foreground_indices] = shifted_foreground[foreground_indices]

        # Normalize and convert to RGB
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Display the frame
        ax.imshow(combined)
        frame = mplfig_to_npimage(fig)
        frames.append(frame)

        # Update progress
        progress_value = int((t / duration) * 100)
        progress.set(progress_value)
        root.update_idletasks()
    
    # Create video from precomputed frames
    def make_frame(t):
        index = int(t * fps)
        return frames[index]

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile("3d_overlay_custom.mp4", fps=fps, codec='libx264', bitrate="5000k")

def start_gui():
    def load_image():
        file_path = select_file()
        if not file_path:
            return

        # Load and crop the image
        img = Image.open(file_path)
        img = np.array(img)
        cropped_img = crop_to_content(img)

        # Generate depth map with progress
        depth_map = generate_depth_map(cropped_img, progress_var)

        # Update the previews
        original_preview_img = Image.fromarray(cropped_img)
        original_preview_img.thumbnail((200, 200))  # Resize for preview
        original_preview = ImageTk.PhotoImage(original_preview_img)

        depth_preview_img = Image.fromarray((depth_map / depth_map.max() * 255).astype(np.uint8))
        depth_preview_img.thumbnail((200, 200))  # Resize for preview
        depth_preview = ImageTk.PhotoImage(depth_preview_img)

        original_label.config(image=original_preview)
        original_label.image = original_preview
        depth_label.config(image=depth_preview)
        depth_label.image = depth_preview

        # Save for later use
        app_data['image'] = cropped_img
        app_data['depth_map'] = depth_map

        # Enable the render button after image is loaded
        render_button.config(state='normal')

    def render_video():
        if 'image' not in app_data or 'depth_map' not in app_data:
            return

        # Get user settings
        fps = fps_var.get()
        duration = duration_var.get()
        speed_multiplier = speed_var.get()
        direction = direction_var.get()

        # Reset progress
        progress_var.set(0)

        # Create layers and render video with progress
        foreground, background = create_layers(app_data['image'], app_data['depth_map'])
        create_animation(foreground, background, app_data['image'].shape[:2], fps, duration, speed_multiplier, direction, progress_var)

        # Set progress to 100 after completion
        progress_var.set(100)

    def update_fps_label(value):
        # Snap to the nearest multiple of 10
        snapped_value = round(float(value) / 10) * 10
        fps_var.set(snapped_value)
        fps_value_label.config(text=f"{int(snapped_value)} FPS")

    def update_duration_label(value):
        duration_value_label.config(text=f"{int(float(value))} sec")

    def update_speed_label(value):
        speed_value_label.config(text=f"Multiplier: {int(float(value))}")

    def set_direction(selected_direction):
        # Update the direction variable
        direction_var.set(selected_direction)
        # Update button styles to indicate selection
        for btn in direction_buttons:
            if btn["text"] == selected_direction.title():
                btn.configure(bootstyle="primary")  # Highlight selected button
            else:
                btn.configure(bootstyle="secondary")

    # Initialize the main application window with ttkbootstrap
    global root
    root = ttkb.Window(themename="darkly")  # Apply a modern theme
    root.title("3D Animation Creator")
    root.geometry("800x800")

    # Make the window non-resizable
    root.resizable(False, False)

    # Create the main frame to contain all the widgets
    main_frame = ttkb.Frame(root, padding=10)
    main_frame.pack(fill='both', expand=True)

    # Configure grid layout for the main frame
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_rowconfigure(1, weight=1)
    main_frame.grid_rowconfigure(2, weight=1)
    main_frame.grid_rowconfigure(3, weight=1)
    main_frame.grid_rowconfigure(4, weight=1)

    # Frame for previews
    preview_frame = ttkb.Frame(main_frame)
    preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # Original and depth map previews
    original_label = ttkb.Label(preview_frame)
    original_label.pack(side='left', padx=5, expand=True)

    depth_label = ttkb.Label(preview_frame)
    depth_label.pack(side='right', padx=5, expand=True)

    # Load image button
    load_button = ttkb.Button(main_frame, text="Load Image", command=load_image, bootstyle="primary", width=30)
    load_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    # Settings Frame
    settings_frame = ttkb.Labelframe(main_frame, text="Settings", padding=10)
    settings_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
    settings_frame.grid_columnconfigure(1, weight=1)

    # FPS setting
    fps_var = IntVar(value=60)
    fps_label = ttkb.Label(settings_frame, text="FPS:")
    fps_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    fps_scale = ttkb.Scale(settings_frame, from_=10, to=120, orient=HORIZONTAL, variable=fps_var, command=update_fps_label)
    fps_scale.grid(row=0, column=1, padx=5, pady=5, sticky='we')
    fps_value_label = ttkb.Label(settings_frame, text=f"{fps_var.get()} FPS")
    fps_value_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

    # Duration setting
    duration_var = IntVar(value=4)
    duration_label = ttkb.Label(settings_frame, text="Duration:")
    duration_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    duration_scale = ttkb.Scale(settings_frame, from_=1, to=10, orient=HORIZONTAL, variable=duration_var, command=update_duration_label)
    duration_scale.grid(row=1, column=1, padx=5, pady=5, sticky='we')
    duration_value_label = ttkb.Label(settings_frame, text=f"{duration_var.get()} sec")
    duration_value_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')

    # Speed setting
    speed_var = IntVar(value=1)
    speed_label = ttkb.Label(settings_frame, text="Speed:")
    speed_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
    speed_scale = ttkb.Scale(settings_frame, from_=1, to=5, orient=HORIZONTAL, variable=speed_var, command=update_speed_label)
    speed_scale.grid(row=2, column=1, padx=5, pady=5, sticky='we')
    speed_value_label = ttkb.Label(settings_frame, text=f"Multiplier: {speed_var.get()}")
    speed_value_label.grid(row=2, column=2, padx=5, pady=5, sticky='w')

    # Animation direction selection with buttons
    direction_label = ttkb.Label(settings_frame, text="Direction:")
    direction_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')

    direction_var = StringVar(value="Horizontal")  # Default value
    direction_options = ["Vertical", "Horizontal", "Circle Clockwise", "Circle Counterclockwise", 
                         "Zoom In", "Zoom Out", "Zoom From Center", "Perspective", 
                         "Diagonal", "Zoom From Corner", "Rotation", "Sway", "Spiral", "Bounce", "Tilt"]

    direction_buttons_frame = ttkb.Frame(settings_frame)
    direction_buttons_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky='we')

    # Create a button for each direction option in a grid layout
    direction_buttons = []
    for i, option in enumerate(direction_options):
        button = ttkb.Button(direction_buttons_frame, text=option.title(), command=lambda opt=option: set_direction(opt), bootstyle="secondary", width=20)
        button.grid(row=i//3, column=i%3, padx=2, pady=2, sticky='we')
        direction_buttons.append(button)

    # Render video button (disabled by default)
    render_button = ttkb.Button(main_frame, text="Render Video", command=render_video, bootstyle="success", state='disabled', width=30)
    render_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    # Progress bar
    progress_frame = ttkb.Frame(main_frame)
    progress_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
    progress_var = IntVar()
    progress_bar = ttkb.Progressbar(progress_frame, orient=HORIZONTAL, length=300, mode='determinate', variable=progress_var)
    progress_bar.pack(fill='x', expand=True)

    # Store data
    app_data = {}

    root.mainloop()

if __name__ == "__main__":
    start_gui()
