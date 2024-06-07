from tkinter import *
from tkinter import ttk

root = Tk()

root.geometry('400x400')

mybutton = ttk.Button(root, text='Hello World!')

mybutton.place(relx=0.5, rely=0.5, anchor=CENTER)

root.mainloop()
