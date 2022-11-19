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
def A(pt: np.ndarray):
    assert len(pt) == 2
    return (pt * np.array([1, -1])) + np.array([window_w/2, window_h/2])


def Ainv(pt: np.ndarray):
    assert len(pt) == 2
    return (pt - np.array([window_w/2, window_h/2])) * np.array([1, -1])


# Quadratic solve
def min_t(a, b, c):
    disc = (b * b) - (4 * a * c)
    denom = 2 * a
    t = -1
    if disc >= 0:
        t = min((-b + np.sqrt(disc)) / denom, (-b - np.sqrt(disc)) / denom)

    return t


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

    def first_end_effector_chain(self, limbs=[]):
        limbs.append(self)
        if len(self.children) == 0:
            return limbs

        return self.children[0].first_end_effector_chain(limbs)

    def __repr__(self):
        return 'Limb #'+str(self.id)


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


# Returns a list of limbs intersecting the sphere of influence, in order of connection.
def limbs_in_soi(chain):
    global collider, soi
    inside = []
    for limb in chain:
        prox_in = np.linalg.norm(limb.proximal() - collider) < soi
        dist_in = np.linalg.norm(limb.distal() - collider) < soi
        if prox_in or dist_in:
            inside.append(limb)
            continue

        unit_prox_to_dist = limb.distal() - limb.proximal(); unit_prox_to_dist /= np.linalg.norm(unit_prox_to_dist)
        center_to_prox = limb.proximal() - collider
        d, f = unit_prox_to_dist, center_to_prox
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - (soi * soi)
        if -1 < min_t(a, b, c) <= limb.limblen:
            inside.append(limb)

    return inside


# Gradient of sphere of influence potential function
def grad_phi(chain, influenced_limbs):
    global collider, soi

    grad = np.zeros((len(chain), 1))
    for limb in influenced_limbs:
        J_i_trans = jacobian(chain, limb.id).T
        mid = 0.5 * (limb.proximal() + limb.distal())
        v = np.array([collider - mid]).T
        grad += 2 * np.dot(J_i_trans, v)

    return -grad


# Minimum and maximum radii of reach
def minmax_radii(limblens):
    maximum = sum(limblens)
    total = maximum
    for length in limblens:
        rest = total - length
        if length > rest:
            return 1.001 * (length - rest), 0.999 * maximum

    return 0, 0.999 * maximum
        

# Build Kinematic Chain
pos = np.array([0, 0])
limblens = [40, 40, 40, 400, 40, 40, 40]  # For uniform: [80 for _ in range(8)]
root_limb = Limb2D(None, 0.0, limblens[0], worldloc=pos)
prev = root_limb
dtheta = np.pi/20
for i in range(1, len(limblens)):
    prev = Limb2D(prev, dtheta, limblens[i])


# IK Params
epochs = 1000
target = np.array([0, sum(limblens)])
minmax = minmax_radii(limblens)
# Avoid params
collider = np.array([-sum(limblens), 0])  # though it's not really a collider, just a point to avoid
soi = 100  # Radius of sphere of influence. Also displayed.
gain = 0.000004

# GUI Parameters
mouse_mode = 0  # 0 = IK target, 1 = Collision object
joint_radius = 7
target_radius = 15
collider_radius = 15  # simply for display, has no influence on algorithm
running = False
assert collider_radius < soi

# Switches
show_soi = False
show_reach = True

# Main runner
def rerun():
    global w, root_limb, target, joint_radius, i, target, target_radius, collider, soi, target_radius,\
        collider_radius, gain, show_soi, epochs, running, minmax, show_reach

    w.configure(background='black')

    for i in range(epochs):
        running = True
        w.delete('all')
        # DRAW 2D DISPLAY ——————————————————————————————————————————————————————————————————————

        # Draw reach circles
        origin = np.array([0, 0])
        w.create_oval(*A(origin - minmax[0]), *A(origin + minmax[0]), fill='', outline='yellow')
        w.create_oval(*A(origin - minmax[1]), *A(origin + minmax[1]), fill='', outline='yellow')

        # Draw collider
        if show_soi:
            w.create_oval(*A(collider - soi), *A(collider + soi), fill='#252400', outline='')
        w.create_oval(*A(collider - collider_radius), *A(collider + collider_radius), fill='yellow', outline='')

        # Draw robot arm
        root_joint_loc = A(root_limb.proximal())
        w.create_oval(*(root_joint_loc - joint_radius), *(root_joint_loc + joint_radius), fill='blue')
        for pair in get_line_points(root_limb):
            p0, p1 = pair[0], pair[1]
            w.create_line(*p0, *p1, fill='red', width=4)
            w.create_oval(*(p1 - joint_radius), *(p1 + joint_radius), fill='blue')

        # Draw target
        w.create_oval(*A(target - target_radius), *A(target + target_radius), fill='green')

        # ———————————————————————————————————————————————————————————————————————————————————————————————
        # MAIN IK ALGORITHM + Nullspace Collision Avoidance
        chain = root_limb.first_end_effector_chain(limbs=[])

        influenced = limbs_in_soi(chain)
        grad_phi(chain, influenced)

        J = jacobian(chain)
        J_pinv = np.linalg.pinv(J)
        JpinvJ = np.dot(J_pinv, J)

        current = chain[-1].distal()
        theta_n = np.array([limb.angle for limb in chain])

        ik_update = np.array([np.dot(J_pinv, (target - current))]).T
        avoid_update = gain * grad_phi(chain, influenced)
        null_op = np.eye(J.shape[1]) - JpinvJ
        update = ik_update + np.dot(null_op, avoid_update)

        theta_nplus1 = np.array([theta_n]).T + update
        for theta, limb in zip(theta_nplus1.T[0], chain):
            limb.set_angle(theta)

        w.update()

    running = False


# Key bind
def key_pressed(event):
    global mouse_mode
    if event.char == 'x':
        mouse_mode = (mouse_mode + 1) % 2


# MOUSE METHODS —————————————————————————————————————————————
def mouse_click(event):
    pass


def mouse_release(event):
    pass


def left_drag(event):
    pass


def motion(event):
    global target, collider, minmax, mouse_mode, running
    screen_pt = Ainv(np.array([event.x, event.y]))
    norm = np.linalg.norm(screen_pt)

    if mouse_mode == 0:
        # If norm is 0, don't update the target.
        if norm != 0:
            hat = screen_pt / norm
            target = min(minmax[1], max(minmax[0], norm)) * hat

    elif mouse_mode == 1:
        collider = screen_pt

    if not running:
        rerun()


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