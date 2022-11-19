# By Aditya Abhyankar, November 2022
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
def A(pt: np.ndarray):
    assert len(pt) == 2
    return (pt * np.array([1, -1])) + np.array([window_w/2, window_h/2])


def Ainv(pt: np.ndarray):
    assert len(pt) == 2
    return (pt - np.array([window_w/2, window_h/2])) * np.array([1, -1])


# Tack on the extra 1 at the end to make it a 3d vector
def three(point:np.ndarray):
    assert len(point) == 2
    return np.append(point, 1)


# Remove extra 1 at the end to make it a 3d vector
def two(point:np.ndarray):
    assert len(point) == 3 and point[2] == 1  # make sure the last entry is a 1!
    return point[:2]


class Limb2D:
    next_id = 0

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

        self.id = Limb2D.next_id
        Limb2D.next_id += 1

    def set_angle(self, angle):
        self.angle = angle
        self.R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.H = np.vstack([np.hstack([self.R, np.array([self.d]).T]), np.array([0, 0, 1])])

    def set_pos(self, d):
        assert self.parent is None
        self.d = d
        self.set_angle(self.angle)

    def local_to_parent(self, point):
        assert len(point) == 3  # must be a homogeneous point
        return np.dot(self.H, point)

    def local_to_world(self, point):
        point = self.local_to_parent(point)
        return self.parent.local_to_world(point) if self.parent is not None else point

    # Returns proximal joint in target frame
    def proximal(self, world=True):
        local = three(np.array([0, 0]))
        tg = self.local_to_world(local) if world else self.local_to_parent(local)
        return two(tg)

    # Returns distal joint in target frame
    def distal(self, world=True):
        local = three(np.array([self.limblen, 0]))
        tg = self.local_to_world(local) if world else self.local_to_parent(local)
        return two(tg)

    def end_effectors(self, ends=[]):
        if len(self.children) == 0:
            ends.append(self)
            return

        for child in self.children:
            child.end_effectors(ends)

        return ends

    def end_pos(self):
        return self.end_effectors()[0].distal()

    # Builds chain starting from this joint to an arbitrary end-effector (grandchild)
    def first_end_effector_chain(self, limbs=[]):
        limbs.append(self)
        if len(self.children) == 0:
            return limbs

        return self.children[0].first_end_effector_chain(limbs)

    def get_chain_to_root(self, include_self=False, max_size=1000):
        '''
            Builds a kinematic chain within a desired 'max_size' towards the root limb.
            We include this limb iff 'include_self' is set to True. Caller should set
            it to be true iff the clicked joint is an end-effector.

            NOTE: Final limb must be the proximal-most limb in the chain (chain closest to root, in graph distance).
        '''
        limbs = []
        if include_self:
            limbs.append(self)

        curr = self.parent
        for i in range(max_size-1):
            if curr is None:
                break
            limbs.append(curr)
            curr = curr.parent

        return limbs

    def __repr__(self):
        return 'Limb #' + str(self.id)


def get_line_points(limb, pts=[]):
    if len(pts) == 0:
        pts = []

    # Returns as many pairs of points as there are limbs
    pts.append(np.array([A(limb.proximal()), A(limb.distal())]))
    for child in limb.children:
        get_line_points(child, pts)

    return pts


# Computes the Jacobian of the joint positions wrt joint angles for the kinematic
# chain leading up to the end effector of the input chain, via DFS starting at the root limb.
def jacobian(chain, wrt=-1):
    '''
        wrt: The int indicating the limb (or # of angles) that the Jacobian is taken with respect to.
             -1 indicates it is for the end effector, thus all the angles are included.

        NOTE: By default, if wrt != -1 and is less than len(chain)-1, the # of columns of the Jacobian
              stays the same, but the block matrices corresponding to limbs that are ancestors of the
              limb in question are all just zeroed out. This is to update whole theta vector in one
              fell swoop without size readjustments.
    '''
    end_eff_pos = chain[wrt].distal()
    col_zero = np.array([[0, 0]]).T
    if wrt == -1:
        wrt = len(chain) - 1
    J = None
    for i, limb in enumerate(chain):
        r = end_eff_pos - limb.proximal()

        # If with respect to end effector, i will never be greater than wrt.
        col = col_zero if i > wrt else np.array([[-r[1], r[0]]]).T
        if J is None:
            J = col
        else:
            J = np.hstack([J, col])

    return J


# Get minimum and maximum reach radii of a given chain
def minmax_radii(chain):
    maximum = 0
    for limb in chain:
        maximum += limb.limblen

    total = maximum
    for limb in chain:
        length = limb.limblen
        rest = total - length
        if length > rest:
            return 1.001 * (length - rest), 0.999 * maximum

    return 0, 0.999 * maximum


# Build Kinematic Chains (3 of them)
# 1. Some params
rootpos = np.array([0, 0])
limblen = 80
# 2. Legs
thigh1 = Limb2D(None, -np.pi/4, limblen, worldloc=rootpos)  # root!
shin1 = Limb2D(thigh1, -np.pi/4, limblen)
foot1 = Limb2D(shin1, np.pi/2, limblen/2.5)
thigh2 = Limb2D(None, -3*np.pi/4, limblen, rootpos)  # root!
shin2 = Limb2D(thigh2, -np.pi/4, limblen)
foot2 = Limb2D(shin2, np.pi/2, limblen/2.5)
# 3. Upper body
torso = Limb2D(None, np.pi/2, limblen*2, rootpos)  # root!
humerous1 = Limb2D(torso, -3*np.pi/4, limblen)
forearm1 = Limb2D(humerous1, np.pi/4, limblen)
humerous2 = Limb2D(torso, np.pi/2, limblen)
forearm2 = Limb2D(humerous2, np.pi/4, limblen)
skull = Limb2D(torso, 0, limblen)
# List of limbs
limbs = [thigh1, shin1, foot1, thigh2, shin2, foot2, torso, humerous1, forearm1, humerous2, forearm2, skull]


# IK Params
n_epochs = 30
chain = []
target = None
min_reach, max_reach = 0, 0

# GUI Paras
joint_radius = 7
click_radius = joint_radius * 1.5
running = True

# Switches
mode = True  # modes: True='ik' or False='fk'
show_joints = False


# Main runner
def rerun():
    global w, torso, thigh1, thigh2, joint_radius, skull, n_epochs, chain, target, running, mode, show_joints,\
           min_reach, max_reach

    # Setup
    w.configure(background='black')
    w.delete('all')

    # Indicate mode
    w.create_text(window_w / 2, window_h - 40, text='IK Mode' if mode else 'FK Mode', fill='red', font='Avenir 30')

    # Solve IK problem (FK problem is just IK problem of chain length 1)
    if len(chain) > 0 and chain[0] is not None and target is not None:
        for i in range(n_epochs):
            J = jacobian(chain)

            J_pinv = np.linalg.pinv(J)

            current = chain[-1].distal()
            update = np.array([np.dot(J_pinv, (target - current))]).T

            theta_n = np.array([limb.angle for limb in chain])
            theta_nplus1 = np.array([theta_n]).T + update
            for theta, limb in zip(theta_nplus1.T[0], chain):
                limb.set_angle(theta)

    # Draw target
    if target is not None:
        # Draw target circle
        center = A(target)
        w.create_oval(*(center - joint_radius), *(center + joint_radius), fill='', outline='yellow')

        # Draw reach circles
        if chain[0] is not None:
            origin = torso.proximal() if mode else chain[0].proximal()
            w.create_oval(*A(origin - min_reach), *A(origin + min_reach), fill='', outline='purple')
            w.create_oval(*A(origin - max_reach), *A(origin + max_reach), fill='', outline='blue')

    # Draw character
    # 1. Draw fake head
    center = A((skull.proximal() + skull.distal()) / 2)
    head_radius = limblen * 0.5
    w.create_oval(*(center - head_radius), *(center + head_radius), fill='white', outline='', width=4)
    # 2. Draw upper body
    root_loc = A(torso.proximal())
    w.create_oval(*(root_loc - joint_radius), *(root_loc + joint_radius), fill='white') if show_joints else None
    for pair in get_line_points(torso):
        p0, p1 = pair[0], pair[1]
        w.create_line(*p0, *p1, fill='white', width=4)
        w.create_oval(*(p1 - joint_radius), *(p1 + joint_radius), fill='white') if show_joints else None
    # 3. Draw legs
    pairs = get_line_points(thigh1)
    pairs.extend(get_line_points(thigh2))
    for pair in pairs:
        p0, p1 = pair[0], pair[1]
        w.create_line(*p0, *p1, fill='white', width=4)
        w.create_oval(*(p1 - joint_radius), *(p1 + joint_radius), fill='white') if show_joints else None

    w.update()
    running = False


# Key bind
def key_pressed(event):
    global mode
    if event.char == ' ':
        mode = not mode

    rerun()


# MOUSE METHODS ————————————————————————————————————————————
def mouse_click(event):
    '''
         Fires when mouse button is first pressed down.
         Here, we'll build the kinematic chain for the IK problem.
    '''
    global click_radius, limbs, chain, torso
    clicked_pt = Ainv(np.array([event.x, event.y]))
    for limb in limbs:
        # Check only the proximal joint of each limb. Unless it's an end-effector, then check it's distal too.
        is_endeff = len(limb.children) == 0
        b1 = np.linalg.norm(limb.proximal() - clicked_pt) < click_radius
        b2 = is_endeff and np.linalg.norm(limb.distal() - clicked_pt) < click_radius
        if b1 or b2:
            # Create IK chain
            if mode:
                chain = limb.get_chain_to_root(include_self=b2, max_size=10000)  # TODO: account for variable max chain size
                chain.reverse()
            # Or create FK "chain".
            else:
                chain = [limb if b2 else limb.parent]  # NOTE: Chain could contain "NONE" in case that a root is selected
            break


def mouse_release(event):
    global chain, target, running
    chain = []
    target = None
    rerun()


def left_drag(event):
    '''
        Set the target based on click position + resolve FKIK problem, and redraw.
    '''
    global target, chain, running, torso, thigh1, thigh2, min_reach, max_reach
    dragged_pt = Ainv(np.array([event.x, event.y]))
    if len(chain) > 0:
        target = dragged_pt
        if chain[0] is None:  # "The" root is selected (there actually are 3 roots)
            torso.set_pos(dragged_pt)
            thigh1.set_pos(dragged_pt)
            thigh2.set_pos(dragged_pt)
            rerun()
        else:
            # Compute max and min reach
            rootpos = chain[0].proximal()
            min_reach, max_reach = minmax_radii(chain)

            # Compute target (If norm is 0 just keep it where it is.)
            norm = np.linalg.norm(target - rootpos)
            if norm != 0:
                hat = (dragged_pt - rootpos) / norm
                target = rootpos + (max(min_reach, min(max_reach, norm)) * hat)

            # 2 hours gone down the drain to figure out that this makes it friggin work
            if not running:
                running = True
                rerun()


def motion(event):
    pass


# Mouse bind
w.bind('<Motion>', motion)
w.bind("<Button-1>", mouse_click)
w.bind("<ButtonRelease-1>", mouse_release)
w.bind('<B1-Motion>', left_drag)

root.bind("<KeyPress>", key_pressed)
w.pack()

# We call the main function
if __name__ == '__main__':
    rerun()

# Necessary line for Tkinter
mainloop()