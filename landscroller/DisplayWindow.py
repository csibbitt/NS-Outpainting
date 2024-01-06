from collections import deque
from PIL import ImageTk
import threading
import time
import tkinter as tk

class DisplayWindow(tk.Toplevel, threading.Thread):
    def __init__(self, container):
        threading.Thread.__init__(self, name='display', daemon=True)
        tk.Toplevel.__init__(self, container)

        self.app = container
        self.app.display_window = self

        # ImageGen and viewport sizes
        self.eval_width = 128
        self.eval_height = 128
        self.scale = 2
        self.patch_height = self.eval_height * self.scale
        self.patch_width = self.eval_width * self.scale
        self.height = self.patch_height
        self.width = 1920
        self.vwidth = self.eval_width * self.app.buffer_size.get() * self.scale

        # Instance vars
        self.last_buffer_size = -1
        self.n = 0
        self.nn = 0
        self.pimg = None
        # These next two are because the very first pull from the model returns two patches
        self.prediction_count = 1
        self.prefetch_count = -1

        # Time related
        self.start_time = time.time()
        self.last_run_time = self.start_time
        self.avg_run_time = self.start_time

        # Model callback and save co-ordination
        self.current_img = None
        self.img_q = deque(maxlen=1)
        self.last_img = None
        self.last_img_lock = threading.Lock()

        self.start()

    def eval_callback(self, img):
        if self.prediction_count == 1:
            label = self.app.control_window.seed_preview_label
            label.image = ImageTk.PhotoImage(img.crop((0,0,128,128)).resize((32,32)))
            label.config(image=label.image)

        simg = ImageTk.PhotoImage(img.resize((img.width * self.scale, img.height * self.scale)))
        while len(self.img_q) >= self.img_q.maxlen:
            time.sleep(0.25)
        self.img_q.append(simg)
        with self.last_img_lock:
            self.last_img = self.current_img
        self.current_img = img
        time.sleep(0.25)

    def run(self):
        self.title("landscroller - Display")
        self.geometry('+0+0')

        canvas = tk.Canvas(self, width=self.width, height=self.height, scrollregion=(0, 0, self.vwidth, self.height))
        canvas.pack(fill="both", expand=True)

        canvas.image = ImageTk.PhotoImage(file='startup.jpg')  # Keep a reference to avoid garbage collection
        self.canvas_image_id = canvas.create_image(0, 0, image=canvas.image, anchor='nw')

        scrollbar_x = tk.Scrollbar(self, orient="horizontal", command=canvas.xview)
        scrollbar_x.pack(side="bottom", fill="x")
        canvas.configure(xscrollcommand=scrollbar_x.set)

        # Prevent closing
        def on_closing():
            pass
        self.protocol("WM_DELETE_WINDOW", on_closing)

        # Wait here for the first image to appear in the queue
        while len(self.img_q) < 1:
            time.sleep(1)

        # Schedule the first update
        self.after(int(1000/self.app.fps.get()), self.update)
        self.wait_window

    def update(self):
        if self.app.paused.get(): self.after(int(250), self.update); return

        buf_siz = self.app.buffer_size.get()

        # Adjust scroll region if canvas size has changed
        new_canvas_vwidth = buf_siz * self.eval_width * self.scale
        if new_canvas_vwidth != self.vwidth:
            self.vwidth = new_canvas_vwidth
            self.children['!canvas'].configure(scrollregion=(0, 0, self.vwidth, self.height))

        # Fetch another image if we've scrolled a full width
        if self.n > self.patch_width:
            self.n = 0; self.prefetch_count += 1

        # Adjust prefetch if buffer size has changed
        if self.last_buffer_size != buf_siz:
            if self.last_buffer_size == -1:
                self.prefetch_count = buf_siz - 1 # Very first prediction is double-wide
            else:
                self.prefetch_count = max(0,  self.prefetch_count + (buf_siz - self.last_buffer_size))
            self.last_buffer_size = buf_siz

        # If we need to fetch more images
        if self.prefetch_count > 0:
            self.get_next_image(buf_siz)

        # Scroll 1px and maintain leading edge
        cw = self.children['!canvas'].winfo_width()
        self.children['!canvas'].xview_moveto( ( ( ((self.vwidth/cw) - 1)/(self.vwidth/cw) * self.vwidth ) - self.patch_width + self.n) / self.vwidth)  # Do you believe in magic?

        # Next update scheduling calculations
        l_fps = self.app.fps.get()
        now = time.time()
        yield_for = int( 1000/l_fps - (now - self.last_run_time) )
        self.after(yield_for, self.update)

        # FPS calculation
        if self.nn % l_fps == 0:
            self.app.avg_fps.set(f'({round(1 / (now - self.avg_run_time) * l_fps)})')
            self.avg_run_time = now
        self.last_run_time = now
        self.app.running_time_str.set(f'{int(time.time() - self.start_time)}s')

        # End of update()
        self.n += 1; self.nn += 1

    def get_next_image(self, buf_siz):
        #print("[D] Update pre q get")
        if len(self.img_q) > 0:
            self.pimg = self.img_q.popleft()
            #print("[D] Update post q get")
            self.prefetch_count -= 1
            self.prediction_count += 1
            self.children['!canvas'].itemconfigure(self.canvas_image_id, image=self.pimg)
            self.app.predictions_string.set(f'{self.prediction_count - buf_siz + self.prefetch_count + 1}-{self.prediction_count}')
            self.app.status.set(f'Got image #{self.prediction_count}')
        else:
            self.app.status.set(f'Waiting on a new images ({self.prefetch_count})...')

    def save_viewport(self):
        prev_pause = self.app.paused.get()
        self.app.paused.set(True)
        name = f'outputs/view-{int(self.start_time)}-{self.app.input_hash.get()}-n{self.nn}-({self.app.predictions_string.get()}).jpg'
        cw = self.children['!canvas'].winfo_width()
        with self.last_img_lock:
            start_col = self.children['!canvas'].xview()[0] * self.last_img.width
            cropped = self.last_img.crop( (start_col, 0, start_col + (cw / self.scale), self.eval_height) )
            cropped.save(name)
            preview_save(cropped, name)
        self.app.status.set('Saved viewport')
        self.app.paused.set(prev_pause)

    def save_buffer(self):
        prev_pause = self.app.paused.get()
        self.app.paused.set(True)
        name = f'outputs/buf-{int(self.start_time)}-{self.app.input_hash.get()}-n{self.nn}-({self.app.predictions_string.get()}).jpg'
        with self.last_img_lock:
            self.last_img.save(name)
            preview_save(self.last_img, name)
        self.app.status.set('Saved buffer')
        self.app.paused.set(prev_pause)

    def shuffle(self):
        self.app.shuffle_flag.set(True)
        self.prefetch_count = self.app.buffer_size.get() + 2 # Empty the queued and buffered images ASAP
        self.prediction_count = -1
        self.nn = -1

def preview_save(img, name):
    pw = tk.Toplevel()
    pw.title('Saved - ' + name)

    screen_width = 1920 # pw.winfo_screenwidth() returns my entire virtual screen
    scale = screen_width / img.width

    canvas = tk.Canvas(pw, width=screen_width, height=img.height * scale)
    canvas.pack(fill="both", expand=True)

    canvas.pimg = ImageTk.PhotoImage(img.resize((int(img.width * scale), int(img.height * scale))))
    canvas.create_image((0, 0), image=canvas.pimg, anchor='nw')