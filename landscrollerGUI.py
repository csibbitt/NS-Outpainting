#!/usr/bin/env python
import argparse
import threading
import run_model as Eval
from landscroller import App, ControlWindow, DisplayWindow

import os
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infinite landscape scroller')
    parser.add_argument('-mock', action='store_true', default=False)

    args = parser.parse_args()

    app = App()
    DisplayWindow(app)
    ControlWindow(app)

    target =  Eval.mainSession if not args.mock else Eval.mainSessionMock
    threading.Thread(name='eval', target=target, args=(app.buffer_size, app.display_window.eval_callback, app.shuffle_flag, app.input_images, app.input_hash, app.mix_strength),  daemon=True).start()

    app.mainloop()