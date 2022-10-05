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


# IMMEDIATE TODO: FIX ANGLE DATA REPRESENTATION!!
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

    def origin(self):
        return self.local_to_world(np.array([0, 0, 1]))

    def origin_(self):
        return self.origin()[:2]


def get_line_points(joint, pts=[]):
    # Returns as many pairs of points as there are limbs
    # NOTE: End-effectors are joints themselves! Keep that in mind when counting limbs.
    for child in joint.children:
        pts.append([joint.origin()[:2], child.origin()[:2]])
        get_line_points(child, pts)

    return pts


# Build arm
dtheta = -np.pi / 20
n_limbs = 5
pos = np.array([window_w/2, window_h/2])
root_joint = Joint2D(None, 0.0, pos)
prev = root_joint
for i in range(n_limbs):
    prev = Joint2D(prev, dtheta, 100)

j = Joint2D(root_joint, np.pi/2, 100)  # should not overlap joint 1!


# Display Parameters
joint_radius = 10


# Main runner
def runstep():
    global w, root_joint, joint_radius

    w.configure(background='black')
    w.delete('all')
    # DRAW 2D DISPLAY ——————————————————————————————————————————————————————————————————————
    origin = root_joint.origin_()
    w.create_oval(*(origin - joint_radius), *(origin + joint_radius), fill='blue')
    for limb in get_line_points(root_joint):
        p0, p1 = limb[0], limb[1]
        w.create_line(*p0, *p1, fill='red')
        w.create_oval(*(p1 - joint_radius), *(p1 + joint_radius), fill='blue')



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