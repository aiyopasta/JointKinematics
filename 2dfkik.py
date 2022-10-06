# By Aditya Abhyankar, September 2022
from tkinter import *
import numpy as np
np.set_printoptions(suppress=True)

# Window size. Note: 1920/2 = 960 will be the width of each half of the display (2D | 3D)
window_w = 1700
window_h = 1000

# Tkinter Setup
root = Tk()
root.title("2D FKIK")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded

w = Canvas(root, width=window_w, height=window_h)


# Coordinate Shift
def A(point:np.ndarray):
    assert len(point) == 2
    point[0] = point[0] + window_w/2
    point[1] = -point[1] + window_h/2
    return point


# Tack on the extra 1 at the end to make it a 3d vector
def three(point:np.ndarray):
    assert len(point) == 2
    return np.append(point, 1)


# Remove extra 1 at the end to make it a 3d vector
def two(point:np.ndarray):
    assert len(point) == 3 and point[2] == 1  # make sure the last entry is a 1!
    return point[:2]


class Limb2D:
    def __init__(self, parent, angle, limblen, worldloc=None):
        '''
            parent: Either another Limb or None if root.
            angle: Angle (radians) that the limb makes wrt to the +xaxis of the parent's reference frame.
                   If it's the root (and thus has no parent), then it's wrt to the world's +xaxis.
            limblen: Length of limb.
            worldloc: Only has a value if the parent is None, i.e. it's the root limb. In that case worldloc is the
                      location of the proximal joint of this limb in the world.

            NOTE: The frame of reference of each limb has: 1) Its +x-axis aligned with the limb and points from
                  the proximal joint to the distal one, 2) its y-axis perpendicular to the x-axis (counterclockwise
                  90 degrees), and 3) is positioned at the location of the proximal joint.

                  Thus self.H is the homogeneous matrix bringing us from the frame of reference of the proximal joint
                  to that of the the parent limb's proximal joint.

            NOTE 2: To have a tree-like structure where there are multiple limbs sticking out of the ROOT joint,
                    just have multiple, separate kinematic chains (i.e. don't include the second one here).
        '''

        self.limblen = limblen
        self.parent = parent
        if parent is not None:
            assert worldloc is None
            self.d = np.array([parent.limblen, 0])
            self.parent.children.append(self)
        else:
            assert worldloc is not None
            self.d = worldloc

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

    # Returns proximal joint in target frame
    def proximal(self, world=True):
        local = three(np.array([0, 0]))
        target = self.local_to_world(local) if world else self.local_to_parent(local)
        return two(target)

    # Returns distal joint in target frame
    def distal(self, world=True):
        local = three(np.array([self.limblen, 0]))
        target = self.local_to_world(local) if world else self.local_to_parent(local)
        return two(target)


def get_line_points(limb, pts=[]):
    # Returns as many pairs of points as there are limbs
    pts.append([A(limb.proximal()), A(limb.distal())])
    for child in limb.children:
        get_line_points(child, pts)

    return pts


# Build Kinematic Chain
pos = np.array([0, 0])
limblen = 50
n_limbs = 10
root_limb = Limb2D(None, 0.0, limblen, worldloc=pos)

prev = root_limb
dtheta = np.pi/20
for i in range(n_limbs-1):
    prev = Limb2D(prev, dtheta, limblen)

# Display Parameters
joint_radius = 10


# Main runner
def runstep():
    global w, root_limb, joint_radius

    w.configure(background='black')
    w.delete('all')
    # DRAW 2D DISPLAY ——————————————————————————————————————————————————————————————————————
    root_joint_loc = A(root_limb.proximal())
    w.create_oval(*(root_joint_loc - joint_radius), *(root_joint_loc + joint_radius), fill='blue')
    for pts in get_line_points(root_limb):
        p0, p1 = pts[0], pts[1]
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