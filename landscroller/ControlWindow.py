from dataset.build_dataset import process_input_image
from os.path import basename
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog as filedialog

class ControlWindow():
    def __init__(self, container):

        self.app = container
        self.app.control_window = self
        self.display_window = self.app.display_window
        tk_vsi = self.app.register(validate_spinbox_input)

        time_label = tk.Label(self.app, text="Running Time:")
        time_label.grid(row=0, column=0, sticky="e")
        mix_strength_label = tk.Label(self.app, text="Seed Stab %:")
        mix_strength_label.grid(row=1, column=0, sticky="e")
        fps_label = tk.Label(self.app, text="FPS:")
        fps_label.grid(row=2, column=0, sticky="e")
        buffer_label = tk.Label(self.app, text="Buffer Size:")
        buffer_label.grid(row=3, column=0, sticky="e")
        save_viewport_button = tk.Button(self.app, text="Save Viewport", command=self.display_window.save_viewport)
        save_viewport_button.grid(row=4, column=0)

        time_value = tk.Label(self.app, textvariable=self.app.running_time_str)
        time_value.grid(row=0, column=1, sticky="w")
        mix_and_preview_frame = tk.Frame(self.app)
        mix_and_preview_frame.grid(row=1, column=1, sticky="w")
        mix_strength_spinbox = tk.Spinbox(mix_and_preview_frame, from_=0, to=100, width=5, textvariable=self.app.mix_strength, validate="key", validatecommand=(tk_vsi, '%P'))
        mix_strength_spinbox.grid(row=0, column=0, sticky="w")
        seed_preview_frame = tk.Frame(mix_and_preview_frame, highlightbackground="black", highlightthickness=1, border=1)
        seed_preview_frame.grid(row=0, column=1)
        self.seed_preview_label = tk.Label(seed_preview_frame, image=ImageTk.PhotoImage(Image.new(mode='RGB', size=(32,32))))
        self.seed_preview_label.grid(row=0, column=0)

        fps_frame = tk.Frame(self.app)
        fps_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        fps_spinbox = tk.Spinbox(fps_frame, from_=0.001, to=240, width=5, textvariable=self.app.fps, validate="key", validatecommand=(tk_vsi, '%P'))
        fps_spinbox.grid(row=0, column=1, sticky="w")
        fps_label = tk.Label(fps_frame, textvariable=self.app.avg_fps)
        fps_label.grid(row=0, column=2, sticky="w")
        buffer_frame = tk.Frame(self.app)
        buffer_frame.grid(row=3, column=1, columnspan=2, sticky="w")
        buffer_spinbox = tk.Spinbox(buffer_frame, from_=1, to=100, width=5, textvariable=self.app.buffer_size, validate="key", validatecommand=(tk_vsi, '%P'))
        buffer_spinbox.grid(row=0, column=1, sticky="w")
        prediction_value = tk.Label(buffer_frame, textvariable=self.app.predictions_string)
        prediction_value.grid(row=0, column=2, sticky="w")
        save_buffer_button = tk.Button(self.app, text="Save Buffer", command=self.display_window.save_buffer)
        save_buffer_button.grid(row=4, column=1)

        pause_button = tk.Button(self.app, text="Pause/Play Updates", command=self.pause_play_updates)
        pause_button.grid(row=0, column=2)
        open_file_button = tk.Button(self.app, text="Open File", command=self.open_file)
        open_file_button.grid(row=1, column=2)
        save_buffer_button = tk.Button(self.app, text="Shuffle", command=self.display_window.shuffle)
        save_buffer_button.grid(row=2, column=2)

        status_frame = tk.Frame(self.app, highlightbackground="black", highlightthickness=1, border=2)
        status_frame.grid(row=5, column=0, columnspan=3, sticky=tk.E + tk.W)
        status_label = tk.Label(status_frame, textvariable=self.app.status)
        status_label.grid()

    def open_file(self):
        label = self.seed_preview_label
        file_types = [("Image Files", "*.bmp *.png *.jpg *.jpeg *.gif *.tif *.tiff"),("All Files", "*"),]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            img, img_hash = process_input_image(file_path)
            in_im = self.app.input_images
            if (len(in_im) == 0):
                    in_im.append(img)
            else:
                in_im[0] = img
            self.app.input_hash.set('f_' + img_hash)
            label.image = ImageTk.PhotoImage(img.resize((32,32)))
            label.config(image=label.image)
            self.app.status.set(f'Loaded {basename(file_path)} ({self.app.input_hash})')
            self.display_window.shuffle()

    def pause_play_updates(self):
        self.app.paused.set(not self.app.paused.get())
        self.app.status.set(f'{"" if self.app.paused.get() else "Un"}Paused')


def validate_spinbox_input(new_value):
    if new_value == '':
        return False
    try:
        return float(new_value) > 0
    except ValueError:
        return False  # Not a valid integer
