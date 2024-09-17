import torch
import cv2
import numpy as np
import os
import time
from tkinter import Tk, filedialog, Scale, HORIZONTAL, IntVar, Canvas, Scrollbar, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from u2net_model import U2NET  # Make sure u2net_model.py is in the same directory

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global data storage
app_data = {}

# Define direction mapping
direction_mapping = {
    "Vertical": "vertical",
    "Horizontal": "horizontal",
    "Circle Clockwise": "circle_clockwise",
    "Circle Counterclockwise": "circle_counterclockwise",
    "Zoom In": "zoom_in",
    "Zoom Out": "zoom_out",
    "Zoom From Center": "zoom_from_center",
    "Perspective": "perspective",
    "Diagonal": "diagonal",
    "Zoom From Corner": "zoom_from_corner",
    "Rotation": "rotation",
    "Sway": "sway",
    "Spiral": "spiral",
    "Bounce": "bounce",
    "Tilt": "tilt",
    "Pokemon Card": "pokemon_card"
}

# Define model type mapping
model_type_mapping = {
    "DPT_Large (default)": "DPT_Large",
    "DPT_Hybrid": "DPT_Hybrid",
    "MiDaS_small": "MiDaS_small",
    "MiDaS": "MiDaS",
    "U2-Net": "U2-Net"
}

# Load the MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

def load_model(model_type, progress=None):
    if progress is not None:
        progress.set(0)
        root.update_idletasks()
    # Load the MiDaS model
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()
    if progress is not None:
        progress.set(100)
        root.update_idletasks()
    return model

def load_u2net_model(progress=None):
    if progress is not None:
        progress.set(0)
        root.update_idletasks()
    
    # Initialize U2NET model
    model = U2NET()  # Make sure U2NET is defined in u2net_model.py
    # Load the state_dict into the model
    model.load_state_dict(torch.load('models/u2net.pth', map_location=device))
    model.to(device)
    model.eval()

    if progress is not None:
        progress.set(100)
        root.update_idletasks()
    return model

def generate_segmentation_map(img, progress):
    progress.set(0)
    root.update_idletasks()
    
    # Preprocess the image for U2-Net
    transform = Compose([
        Resize((320, 320)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict segmentation map using U2-Net
    with torch.no_grad():
        prediction = app_data['u2net_model'](input_tensor)
    
    # U2-Net outputs multiple maps, use the most detailed output
    segmentation_map = torch.sigmoid(prediction[0][0]).cpu().numpy()
    segmentation_map = (segmentation_map > 0.5).astype(np.uint8)

    progress.set(100)
    root.update_idletasks()
    
    return segmentation_map

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

    # Predict the depth using the model from app_data
    with torch.no_grad():
        prediction = app_data['model'](input_batch)

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

def create_layers_using_segmentation(image, segmentation_map):
    # Separate foreground and background using the segmentation map
    foreground = cv2.bitwise_and(image, image, mask=segmentation_map)
    background = cv2.inpaint(image, segmentation_map, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return foreground, background

def render_video():
    if 'image' not in app_data or 'depth_map' not in app_data:
        return

    # Get user settings
    fps = fps_var.get()
    duration = duration_var.get()
    speed_multiplier = speed_var.get()
    direction = direction_var.get()
    model_type = app_data['model_type']
    apply_anaglyph = anaglyph_var.get()

    # Get the name of the input file without the extension
    input_file_name = os.path.splitext(os.path.basename(app_data['input_file_path']))[0]

    # Create a unique identifier using the current timestamp
    unique_id = int(time.time())

    # Reset progress
    progress_var.set(0)

    # Create layers and render video with progress
    foreground, background = create_layers(app_data['image'], app_data['depth_map'])
    
    # Pass the file name and other parameters to create_animation
    create_animation(foreground, background, app_data['image'].shape[:2], fps, duration, speed_multiplier, direction, 
                     input_file_name, model_type, unique_id, progress_var, apply_anaglyph)

    # Set progress to 100 after completion
    progress_var.set(100)

def create_animation(foreground, background, image_size, fps, duration, speed_multiplier, direction, 
                     input_file_name, model_type, unique_id, progress, apply_anaglyph_filter=False):
    fig, ax = plt.subplots(figsize=(image_size[1] / 100, image_size[0] / 100), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Precompute frames
    frames = []
    motion_amplitude = 10  # Base amplitude for movement
    motion_speed_factor = speed_multiplier * (4 / duration)  # Speed adjustment based on duration and multiplier

    for t in np.linspace(0, duration, int(fps * duration)):
        ax.clear()
        ax.axis('off')

        # Initialize shifts
        shift_x = 0
        shift_y = 0

        # Default values for shifted_foreground and shifted_background
        shifted_foreground = foreground.copy()
        shifted_background = background.copy()

        # Determine motion based on selected direction
        if direction == "vertical":
            shift_y = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
        elif direction == "horizontal":
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
        elif direction == "circle_clockwise":
            # Circular motion using sine and cosine functions
            shift_x = int(motion_amplitude * np.cos(2 * np.pi * t * motion_speed_factor))
            shift_y = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
        elif direction == "circle_counterclockwise":
            # Counterclockwise motion using negative sine and cosine
            shift_x = int(motion_amplitude * np.cos(-2 * np.pi * t * motion_speed_factor))
            shift_y = int(motion_amplitude * np.sin(-2 * np.pi * t * motion_speed_factor))
        elif direction == "zoom_in":
            # Zoom effect with scaling in
            scale = 1 + 0.5 * (1 + np.sin(2 * np.pi * t * motion_speed_factor))  # Oscillates between 1 and 1.5
            center = (foreground.shape[1]//2, foreground.shape[0]//2)
            M_zoom = cv2.getRotationMatrix2D(center, 0, scale)
            shifted_foreground = cv2.warpAffine(foreground, M_zoom, (foreground.shape[1], foreground.shape[0]))
        elif direction == "zoom_out":
            # Zoom effect with scaling out
            scale = 1 + 0.5 * (1 - np.sin(2 * np.pi * t * motion_speed_factor))  # Oscillates between 1 and 1.5
            center = (foreground.shape[1]//2, foreground.shape[0]//2)
            M_zoom = cv2.getRotationMatrix2D(center, 0, 1/scale)
            shifted_foreground = cv2.warpAffine(foreground, M_zoom, (foreground.shape[1], foreground.shape[0]))
        elif direction == "zoom_from_center":
            scale = 1 + 0.5 * np.sin(2 * np.pi * t * motion_speed_factor)
            center = (foreground.shape[1]//2, foreground.shape[0]//2)
            M_zoom = cv2.getRotationMatrix2D(center, 0, scale)
            shifted_foreground = cv2.warpAffine(foreground, M_zoom, (foreground.shape[1], foreground.shape[0]))
        elif direction == "perspective":
            pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
            pts2 = pts1 + np.float32([[10 * np.sin(2 * np.pi * t * motion_speed_factor)]*2,
                                      [20 * np.sin(2 * np.pi * t * motion_speed_factor)]*2,
                                      [30 * np.sin(2 * np.pi * t * motion_speed_factor)]*2,
                                      [40 * np.sin(2 * np.pi * t * motion_speed_factor)]*2])
            M_persp = cv2.getPerspectiveTransform(pts1, pts2)
            shifted_foreground = cv2.warpPerspective(foreground, M_persp, (foreground.shape[1], foreground.shape[0]))
        elif direction == "diagonal":
            shift_x = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
            shift_y = int(motion_amplitude * np.sin(2 * np.pi * t * motion_speed_factor))
        elif direction == "zoom_from_corner":
            scale = 1 + 0.5 * np.sin(2 * np.pi * t * motion_speed_factor)
            M_zoom = cv2.getRotationMatrix2D((0, 0), 0, scale)
            shifted_foreground = cv2.warpAffine(foreground, M_zoom, (foreground.shape[1], foreground.shape[0]))
        elif direction == "rotation":
            angle = 20 * np.sin(2 * np.pi * t * motion_speed_factor)
            center = (foreground.shape[1]//2, foreground.shape[0]//2)
            M_rotate = cv2.getRotationMatrix2D(center, angle, 1)
            shifted_foreground = cv2.warpAffine(foreground, M_rotate, (foreground.shape[1], foreground.shape[0]))
        elif direction == "sway":
            shift_x = int(motion_amplitude * np.sin(4 * np.pi * t * motion_speed_factor))  # Double speed
        elif direction == "bounce":
            shift_y = int(motion_amplitude * np.abs(np.sin(4 * np.pi * t * motion_speed_factor)))  # Absolute value
        elif direction == "tilt":
            angle = 5 * np.sin(2 * np.pi * t * motion_speed_factor)
            center = (foreground.shape[1]//2, foreground.shape[0]//2)
            M_tilt = cv2.getRotationMatrix2D(center, angle, 1)
            shifted_foreground = cv2.warpAffine(foreground, M_tilt, (foreground.shape[1], foreground.shape[0]))
        elif direction == "spiral":
            angle = 20 * np.sin(2 * np.pi * t * motion_speed_factor)
            scale = 1 + 0.5 * np.sin(2 * np.pi * t * motion_speed_factor)
            center = (foreground.shape[1]//2, foreground.shape[0]//2)
            M_spiral = cv2.getRotationMatrix2D(center, angle, scale)
            shifted_foreground = cv2.warpAffine(foreground, M_spiral, (foreground.shape[1], foreground.shape[0]))
        elif direction == 'pokemon_card':
            # Create a tilt effect by changing perspective
            tilt_angle = np.sin(2 * np.pi * t * motion_speed_factor) * 20  # Tilt angle oscillates
            h, w = foreground.shape[:2]

            # Perspective transformation points
            center_shift = tilt_angle / 2
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([[0, h * 0.05 + center_shift], [w, h * 0.05 - center_shift], 
                               [0, h * 0.95 - center_shift], [w, h * 0.95 + center_shift]])
            
            # Apply perspective transformation
            M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
            tilted_foreground = cv2.warpPerspective(foreground, M_perspective, (w, h))
            
            # Apply color shift to simulate holographic effect
            color_shift = 20 * np.sin(4 * np.pi * t * motion_speed_factor)
            tilted_foreground = cv2.convertScaleAbs(tilted_foreground, alpha=1, beta=color_shift)

            # Adjust brightness for reflection effect
            reflection = cv2.addWeighted(tilted_foreground, 0.7, background, 0.3, 0)

            shifted_foreground = reflection

        # Apply shifting if direction involves linear movement or circle
        if direction in ["vertical", "horizontal", "diagonal", "sway", "bounce", "circle_clockwise", "circle_counterclockwise"]:
            # Shift foreground and background separately
            M_foreground = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted_foreground = cv2.warpAffine(foreground, M_foreground, (foreground.shape[1], foreground.shape[0]))

            M_background = np.float32([[1, 0, -shift_x / 2], [0, 1, -shift_y / 2]])
            shifted_background = cv2.warpAffine(background, M_background, (background.shape[1], background.shape[0]))

        # Apply anaglyph filter if enabled
        if apply_anaglyph_filter:
            shift_amount = 5  # Define a small shift amount for the 3D effect
            anaglyph_foreground = np.zeros_like(shifted_foreground)

            # Create the red-cyan anaglyph effect
            # Left image (red channel)
            anaglyph_foreground[:, :, 0] = np.roll(shifted_foreground[:, :, 0], shift_amount, axis=1)

            # Right image (cyan channel)
            anaglyph_foreground[:, :, 1] = np.roll(shifted_foreground[:, :, 1], -shift_amount, axis=1)
            anaglyph_foreground[:, :, 2] = np.roll(shifted_foreground[:, :, 2], -shift_amount, axis=1)

            shifted_foreground = anaglyph_foreground

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

    # Construct the output file name
    output_file_name = f"{input_file_name}_{model_type}_{direction}_{unique_id}.mp4"
    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(output_file_name, fps=fps, codec='libx264', bitrate="5000k")

def start_gui():
    def load_image():
        file_path = select_file()
        if not file_path:
            return

        # Show a loading indicator
        original_label.config(text="Loading...", image='', compound='center')
        depth_label.config(text="Loading...", image='', compound='center')
        root.update_idletasks()

        # Load and crop the image
        img = Image.open(file_path).convert("RGB")  # Convert to RGB if needed
        img = np.array(img)
        cropped_img = crop_to_content(img)

        if model_type_var.get() == "U2-Net":
            segmentation_map = generate_segmentation_map(cropped_img, progress_var)
            app_data['segmentation_map'] = segmentation_map
            foreground, background = create_layers_using_segmentation(cropped_img, segmentation_map)
            
            # Convert segmentation map to image for preview
            segmentation_preview_img = Image.fromarray((segmentation_map * 255).astype(np.uint8))
            segmentation_preview_img.thumbnail((250, 250))  # Resize for preview
            depth_preview = ImageTk.PhotoImage(segmentation_preview_img)
            
            depth_label.config(image=depth_preview, text='', compound='center')
            depth_label.image = depth_preview
            app_data['depth_preview_img'] = segmentation_preview_img
        else:
            # Generate depth map with progress
            depth_map = generate_depth_map(cropped_img, progress_var)
            app_data['depth_map'] = depth_map
            foreground, background = create_layers(cropped_img, depth_map)

            # Update the depth map preview
            depth_preview_img = Image.fromarray((depth_map / depth_map.max() * 255).astype(np.uint8))
            depth_preview_img.thumbnail((250, 250))  # Resize for preview
            depth_preview = ImageTk.PhotoImage(depth_preview_img)

            depth_label.config(image=depth_preview, text='', compound='center')
            depth_label.image = depth_preview
            app_data['depth_preview_img'] = depth_preview_img

        # Update the original image preview
        original_preview_img = Image.fromarray(cropped_img)
        original_preview_img.thumbnail((250, 250))  # Resize for preview
        original_preview = ImageTk.PhotoImage(original_preview_img)

        original_label.config(image=original_preview, text='', compound='center')
        original_label.image = original_preview
        app_data['image'] = cropped_img
        app_data['original_preview_img'] = original_preview_img
        app_data['input_file_path'] = file_path

        # Enable the render button after image is loaded
        render_button.config(state='normal')

    def save_original_image(event):
        if 'original_preview_img' in app_data:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
            if file_path:
                app_data['original_preview_img'].save(file_path)

    def save_depth_image(event):
        if 'depth_preview_img' in app_data:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
            if file_path:
                app_data['depth_preview_img'].save(file_path)

    def render_video():
        if 'image' not in app_data or 'depth_map' not in app_data:
            return

        # Get user settings
        fps = fps_var.get()
        duration = duration_var.get()
        speed_multiplier = speed_var.get()
        direction = direction_var.get()
        model_type = app_data['model_type']
        apply_anaglyph = anaglyph_var.get()

        # Get the name of the input file without the extension
        input_file_name = os.path.splitext(os.path.basename(app_data['input_file_path']))[0]

        # Create a unique identifier using the current timestamp
        unique_id = int(time.time())

        # Reset progress
        progress_var.set(0)

        # Create layers and render video with progress
        foreground, background = create_layers(app_data['image'], app_data['depth_map'])
        
        # Pass the file name and other parameters to create_animation
        create_animation(foreground, background, app_data['image'].shape[:2], fps, duration, speed_multiplier, direction, 
                         input_file_name, model_type, unique_id, progress_var, apply_anaglyph)

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
        # Map displayed option to internal value
        internal_direction = direction_mapping[selected_direction]
        # Update the direction variable
        direction_var.set(internal_direction)
        # Update button styles to indicate selection
        for btn in direction_buttons:
            if btn["text"] == selected_direction:
                btn.configure(bootstyle="primary")  # Highlight selected button
            else:
                btn.configure(bootstyle="secondary")

    def set_model_type(selected_model_type):
        # Map displayed option to internal model_type string
        if selected_model_type == "U2-Net":
            app_data['u2net_model'] = load_u2net_model(progress_var)
        else:
            internal_model_type = model_type_mapping[selected_model_type]
            app_data['model_type'] = internal_model_type
            app_data['model'] = load_model(internal_model_type, progress_var)

        # Update the model_type variable
        model_type_var.set(selected_model_type)

        # Show loading skeleton in depth map preview
        depth_label.config(text="Generating depth map...", image='', compound='center')
        root.update_idletasks()

        # If U2-Net is not selected, use the default loading process
        if selected_model_type != "U2-Net":
            progress_var.set(0)
            root.update_idletasks()
            app_data['model'] = load_model(internal_model_type, progress_var)

            # Regenerate the depth map if an image is loaded
            if 'image' in app_data:
                depth_map = generate_depth_map(app_data['image'], progress_var)
                app_data['depth_map'] = depth_map

                # Update the depth map preview
                depth_preview_img = Image.fromarray((depth_map / depth_map.max() * 255).astype(np.uint8))
                depth_preview_img.thumbnail((250, 250))  # Resize for preview
                depth_preview = ImageTk.PhotoImage(depth_preview_img)
                depth_label.config(image=depth_preview, text='', compound='center')
                depth_label.image = depth_preview
                app_data['depth_preview_img'] = depth_preview_img  # Update stored image

        # Update button styles to indicate selected model type
        for btn in model_type_buttons:
            if btn["text"] == selected_model_type:
                btn.configure(bootstyle="primary")  # Highlight selected button
            else:
                btn.configure(bootstyle="secondary")

    global root
    root = ttkb.Window(themename="darkly")
    root.title("3D Animation Creator")
    root.minsize(800, 600)  # Set a minimum window size

    # Set the application icon using the .ico file
    # root.iconbitmap('logo.ico')

    # Set the application icon
    icon_image = Image.open('logo.png')
    icon_photo = ImageTk.PhotoImage(icon_image)
    root.iconphoto(False, icon_photo)

    # Allow the window to be resizable
    root.resizable(True, True)

    # Progress variable
    progress_var = IntVar()

    # Initialize model_type and model
    model_type_var = StringVar(value="DPT_Large")
    app_data['model_type'] = "DPT_Large"
    app_data['model'] = load_model(app_data['model_type'], progress_var)

    # Create the main frame to contain all the widgets
    main_frame = ttkb.Frame(root, padding=10)
    main_frame.pack(fill='both', expand=True)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=0)  # Logo frame
    main_frame.grid_rowconfigure(1, weight=1)  # Preview frame
    main_frame.grid_rowconfigure(2, weight=0)  # Load image button
    main_frame.grid_rowconfigure(3, weight=0)  # Model type buttons
    main_frame.grid_rowconfigure(4, weight=1)  # Settings frame
    main_frame.grid_rowconfigure(5, weight=0)  # Render video button
    main_frame.grid_rowconfigure(6, weight=0)  # Progress bar

    # Frame for the logo
    logo_frame = ttkb.Frame(main_frame)
    logo_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
    
    # Display the logo
    # logo_label = ttkb.Label(logo_frame, image=icon_photo)
    # logo_label.image = icon_photo  # Keep a reference to avoid garbage collection
    # logo_label.pack()

    # Frame for previews
    preview_frame = ttkb.Frame(main_frame)
    preview_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
    preview_frame.grid_columnconfigure(0, weight=1)
    preview_frame.grid_columnconfigure(1, weight=1)
    preview_frame.grid_rowconfigure(0, weight=1)

    # Original and depth map placeholders
    original_label = ttkb.Label(preview_frame, text="Original Image Placeholder", anchor='center')
    original_label.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

    depth_label = ttkb.Label(preview_frame, text="Depth Map Placeholder", anchor='center')
    depth_label.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

    # Center the placeholders
    preview_frame.grid_columnconfigure(0, weight=1)
    preview_frame.grid_columnconfigure(1, weight=1)
    preview_frame.grid_rowconfigure(0, weight=1)

    # Bind click events to the labels
    original_label.bind("<Button-1>", save_original_image)
    depth_label.bind("<Button-1>", save_depth_image)

    # Load image button
    load_button = ttkb.Button(main_frame, text="Load Image", command=load_image, bootstyle="primary")
    load_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

    # Model Type variable and options
    model_type_var = StringVar(value="DPT_Large")  # Default value
    model_type_options = list(model_type_mapping.keys())

    # Model type buttons frame, now in main_frame at row=3
    model_type_buttons_frame = ttkb.Frame(main_frame)
    model_type_buttons_frame.grid(row=3, column=0, padx=10, pady=10, sticky='ew')
    model_type_buttons_frame.grid_columnconfigure(tuple(range(len(model_type_options))), weight=1)

    # Create a button for each model type option
    model_type_buttons = []
    for i, option in enumerate(model_type_options):
        button = ttkb.Button(
            model_type_buttons_frame,
            text=option,
            command=lambda opt=option: set_model_type(opt),
            bootstyle="secondary"
        )
        button.grid(row=0, column=i, padx=2, pady=2, sticky='nsew')
        model_type_buttons.append(button)

    # Set default model type
    set_model_type("DPT_Large (default)")

    # Settings Frame
    settings_frame = ttkb.Labelframe(main_frame, text="Settings", padding=10)
    settings_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
    settings_frame.grid_columnconfigure(1, weight=1)
    settings_frame.grid_columnconfigure(2, weight=0)
    settings_frame.grid_rowconfigure(5, weight=1)  # For direction buttons

    # FPS setting
    fps_var = IntVar(value=60)
    fps_label = ttkb.Label(settings_frame, text="FPS:")
    fps_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    fps_scale = ttkb.Scale(settings_frame, from_=10, to=120, orient=HORIZONTAL, variable=fps_var, command=update_fps_label)
    fps_scale.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
    fps_value_label = ttkb.Label(settings_frame, text=f"{fps_var.get()} FPS")
    fps_value_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

    # Duration setting
    duration_var = IntVar(value=4)
    duration_label = ttkb.Label(settings_frame, text="Duration:")
    duration_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    duration_scale = ttkb.Scale(settings_frame, from_=1, to=10, orient=HORIZONTAL, variable=duration_var, command=update_duration_label)
    duration_scale.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
    duration_value_label = ttkb.Label(settings_frame, text=f"{duration_var.get()} sec")
    duration_value_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')

    # Speed setting
    speed_var = IntVar(value=1)
    speed_label = ttkb.Label(settings_frame, text="Speed:")
    speed_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
    speed_scale = ttkb.Scale(settings_frame, from_=1, to=5, orient=HORIZONTAL, variable=speed_var, command=update_speed_label)
    speed_scale.grid(row=2, column=1, padx=5, pady=5, sticky='ew')
    speed_value_label = ttkb.Label(settings_frame, text=f"Multiplier: {speed_var.get()}")
    speed_value_label.grid(row=2, column=2, padx=5, pady=5, sticky='w')

    # Animation direction selection with buttons
    direction_label = ttkb.Label(settings_frame, text="Direction:")
    direction_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')

    # Direction variable and options
    direction_var = StringVar(value=direction_mapping["Horizontal"])  # Default value
    direction_options = list(direction_mapping.keys())

    direction_buttons_frame = ttkb.Frame(settings_frame)
    direction_buttons_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky='nsew')
    direction_buttons_frame.grid_columnconfigure(tuple(range(3)), weight=1)
    direction_buttons_frame.grid_rowconfigure(tuple(range((len(direction_options) + 2) // 3)), weight=1)

    # Create a button for each direction option in a grid layout
    direction_buttons = []
    for i, option in enumerate(direction_options):
        button = ttkb.Button(
            direction_buttons_frame,
            text=option,
            command=lambda opt=option: set_direction(opt),
            bootstyle="secondary"
        )
        button.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky='nsew')
        direction_buttons.append(button)

    # Set default direction
    set_direction("Horizontal")

    # Anaglyph filter option
    anaglyph_var = IntVar()
    anaglyph_checkbox = ttkb.Checkbutton(settings_frame, text="Apply Anaglyph Filter", variable=anaglyph_var)
    anaglyph_checkbox.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    # Render video button (disabled by default)
    render_button = ttkb.Button(main_frame, text="Render Video", command=render_video, bootstyle="success", state='disabled')
    render_button.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

    # Progress bar
    progress_frame = ttkb.Frame(main_frame)
    progress_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
    progress_bar = ttkb.Progressbar(progress_frame, orient=HORIZONTAL, length=300, mode='determinate', variable=progress_var)
    progress_bar.pack(fill='x', expand=True)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
