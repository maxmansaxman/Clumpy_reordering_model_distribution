#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:26:24 2018

@author: Max
"""

import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()

#image = Image.open("Model_T_t_path_gif.gif")
#photo = ImageTk.PhotoImage(image)

#logo = tk.PhotoImage(file = "Model_T_t_path_gif.gif")
#text_image = "this image is a pdf"

w1 = tk.Label(root, image = logo).pack(side="right")
w2 = tk.Label(root, justify = tk.LEFT, padx = 10,
              text = text_image).pack(side = "left")

root.mainloop()