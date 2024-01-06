import tkinter as tk

canvas_width = 1920
patch_width = 256

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("landscroller - Control")
        self.resizable(False,False)

        self.avg_fps = tk.StringVar(value='(0)')
        self.buffer_size = tk.IntVar(value=int(canvas_width / patch_width) + 2)
        self.fps = tk.IntVar(value=24)
        self.input_hash = tk.StringVar(value='0s')
        self.input_images = []
        self.mix_strength = tk.IntVar(value=10)
        self.paused = tk.BooleanVar(value=False)
        self.predictions_string = tk.StringVar(value='0-0')
        self.running_time_str = tk.StringVar(value='0s')

        self.status = tk.StringVar(value='Starting up...')
        self.shuffle_flag = tk.BooleanVar(value=False)

        self.display_window = None
        self.control_window = None
