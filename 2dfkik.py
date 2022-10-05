# By Aditya Abhyankar, September 2022
from tkinter import *
import numpy as np

# Window size. Note: 1920/2 = 960 will be the width of each half of the display (2D | 3D)
window_w = 1700
window_h = 1000

# Tkinter Setup
root = Tk()
root.title("2D FKIK")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded

w = Canvas(root, width=window_w, height=window_h)


class Joint2D:
    def __init__(self, parent, angle, limblen):
        '''
            parent: Either another Joint or None if root.
            angle: Angle (radians) that the limb extending out from this joint will make with the limb
                   connecting the parent to this one. If this is the root, this is just wrt to the
                   world +x-axis.
            limblen: Length of limb from FROM THE PARENT OF THE JOINT TO THIS ONE. If this is the root,
                     this is just the displacement of the joint's frame of reference wrt the world origin.
        '''

        self.parent = parent
        if parent is not None:
            self.d = np.array([limblen * np.cos(parent.angle), limblen * np.sin(parent.angle)])
            self.parent.children.append(self)
        else:
            assert isinstance(limblen, np.ndarray)  # the lazy approach, asking the user to handle special case :)
            self.d = limblen

        self.angle, self.R, self.H = None, None, None
        self.set_angle(angle)
        self.children = []

    def set_angle(self, angle):
        self.angle = angle
        self.R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.H = np.vstack([np.hstack([self.R, np.array([self.d]).T]), np.array([0, 0, 1])])

    def local_to_parent(self, point):
        assert len(point) == 3  # must be a homogeneous point
        return np.dot(self.H, point)

    def local_to_world(self, point):
        point = self.local_to_parent(point)
        return self.parent.local_to_world(point) if self.parent is not None else point


# Build skeleton



# Main runner
def runstep():
    global w

    w.configure(background='black')
    w.delete('all')
    # DRAW 2D DISPLAY ——————————————————————————————————————————————————————————————————————

    # ———————————————————————————————————————————————————————————————————————————————————————————————
    # MAIN ALGORITHM
    w.update()


# Key bind
def key_pressed(event):
    pass


root.bind("<KeyPress>", key_pressed)
w.pack()

# We call the main function
if __name__ == '__main__':
    runstep()

# Necessary line for Tkinter
mainloop()