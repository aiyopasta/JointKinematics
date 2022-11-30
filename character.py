# TODO:
# 1. Do correct interpolation of angles.
# 2. Animate motion cycles. Walk, ducked walk, run, idle.
# 3. Activate 'main_mode' and blend them together based on keyboard input.
# 4. Try adding stuff other than just gaits. Jumping, wall-jump, rolling.


# By Aditya Abhyankar, November 2022
from tkinter import *
from tkinter.messagebox import askyesno
import numpy as np
import copy
import time

np.set_printoptions(suppress=True)

# Window size. Note: 1920/2 = 960 will be the width of each half of the display (2D | 3D)
window_w = 1700
window_h = 1000

# Tkinter Setup
root = Tk()
root.title("2D FKIK")
# root.attributes("-topmost", True)
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


# Simple gait class
class MotionCycle:
    default_state = np.array([-np.pi / 4, -np.pi / 4, np.pi / 2, -3 * np.pi / 4, -np.pi / 4, np.pi / 2,  # Lower body
                              np.pi / 2, -3 * np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 4, 0,             # Upper body
                              0, limblen * 2])                                                   # root pos wrt guide

    def __init__(self, name, duration, keyframes=[], tension=0.5):
        '''
            name: String
            duration: Length of time (floating point) the animation will take place. Doesn't correspond to actual
                      time it'll take in seconds, it only affects the parameterization. Actual time depends on the
                      time.sleep() thingy you use for displaying canvas.
            keyframes: A list of lists of size 2 that contain: a time in [0, 1], and a list of 14 numbers:
                       The first 12 are the LOCAL angles each of the 12 limbs make, and the final
                       2 is the (x, y) position of the root, with respect to a local, "guide joint".

                       [thigh1, shin1, foot1, thigh2, shin2, foot2,
                        torso, humerous1, forearm1, humerous2, forearm2, skull,
                        root_x, root_y]

                       NOTE 1: Guide joint will be located at the x-value of the 3 root nodes, and some fixed y value.
                       NOTE 2: There always MUST be keyframes at times 0 and 1.

            tension: Tension parameter of the cardinal spline interpolating the keyframes.
        '''
        self.name = name
        self.duration = duration
        self.keyframes = keyframes
        if len(self.keyframes) < 2:
            self.keyframes = [[0.0, copy.copy(MotionCycle.default_state)],
                              [1.0, copy.copy(MotionCycle.default_state)]]
        self.__reclamp()

        self.tension = tension
        assert self.keyframes[0][0] == 0.0
        assert self.keyframes[-1][0] == 1.0

    # DO NOT CALL THIS FUNCTION FROM OUTSIDE.
    def __update_key(self, idx: int, new_t: float, new_state: np.ndarray):
        assert 0 <= idx < len(self.keyframes)
        assert 0 <= new_t <= 1 and len(new_state) == 14
        self.keyframes[idx][0] = new_t
        self.keyframes[idx][1] = new_state
        self.__reclamp()

    # CALL THESE INSTEAD
    def update_t(self, idx: int, new_t: float):
        if idx not in {0, len(self.keyframes)-1, -1}:
            self.__update_key(idx, new_t, self.keyframes[idx][1])
        else:
            min_t = 10.0 / (window_w * 0.8)  # derived from the "epsilon" parameter in left_drag() function.
            max_t = 1.0 - min_t
            check1 = self.keyframes[-2][0] < max_t
            check2 = self.keyframes[1][0] > min_t
            if idx in {len(self.keyframes)-1, -1}:
                remaining_val = new_t
                if (new_t > 1.0 and check1) or (new_t < 1.0 and check2):
                    for key in self.keyframes:
                        if key[0] not in {0.0, 1.0}:
                            key[0] = min(max(key[0] * remaining_val, min_t), max_t)

            if idx == 0:
                remaining_val = 1.0 - new_t
                if (new_t > 0.0 and check1) or (new_t < 0.0 and check2):
                    for key in self.keyframes:
                        if key[0] not in {0.0, 1.0}:
                            key[0] = min(max((key[0] * remaining_val) + new_t, min_t), max_t)

    def update_state(self, idx: int, new_state: np.ndarray):
        self.__update_key(idx, self.keyframes[idx][0], np.array(new_state))

    # DON'T CALL THIS FUNCTION FROM OUTSIDE
    def __reclamp(self):
        for i in range(1, len(self.keyframes)):
            prev_vals = self.keyframes[i-1][1]
            vals = self.keyframes[i][1]
            for j in range(len(vals)-2):  # excluding the root positions (last 2 values)
                while abs(vals[j] - prev_vals[j]) >= np.pi:
                    vals[j] -= np.sign(vals[j] - prev_vals[j]) * 2.0 * np.pi

    # Inserts a keyframe at any specified time.
    def insert_key(self, t: float):
        assert 0 <= t <= 1
        if t == 0.0 or t == 1.0:
            self.insert_key_begin_end(t == 1.0)
        else:
            insert_idx = 0
            for idx, key in enumerate(self.keyframes):
                if idx == 0:
                    continue

                min_t, max_t = self.keyframes[idx-1][0], key[0]
                assert t not in {min_t, max_t}  # make sure new keyframe not overlapping existing
                if min_t < t < max_t:
                    insert_idx = idx            # insert at the maximal endpoint of interval, so it's in between

            prev_val = copy.copy(self.keyframes[insert_idx - 1][1])
            self.keyframes.insert(insert_idx, [t, prev_val])

    # Adds a new keyframe either at the beginning or end.
    def insert_key_begin_end(self, end=True):
        '''
            end: False = add key at the beginning, True = add key at the end.

            NOTE 1: ONLY IF you add a keyframe at the end or beginning, all the other ones will be shifted while
            maintaining their proportional distances... while maintaining that the other end is still pinned!

            NOTE 2: The workflow will be such that if you add a new frame at the end, and you try dragging that
            frame on the timeline, it won't work but all the OTHER frames will scoot closer or farther.
        '''
        n_frames = len(self.keyframes) + 1

        # ENDING CASE
        if end:
            # Shift the original last frame down by fixed value, and shift the others down proportionally.
            remaining_len = 1.0 - (1.0 / (n_frames - 1))
            for key in self.keyframes:
                key[0] *= remaining_len                    # notice how time = 0.0 remains there
            # Insert new keyframe
            prev_val = copy.copy(self.keyframes[-1][1])
            self.keyframes.append([1.0, prev_val])

        # BEGINNING CASE
        else:
            # Shift the original first frame up by fixed value, and shift the others up proportionally.
            shift = 1.0 / (n_frames - 1)
            remaining_len = 1.0 - shift
            for key in self.keyframes:
                key[0] = (key[0] * remaining_len) + shift  # notice how time = 1.0 remains there.
            self.keyframes[-1][0] = 1.0                    # but we set it explicitly just in case
            # Insert new keyframe
            prev_val = copy.copy(self.keyframes[0][1])
            self.keyframes.insert(0, [0.0, prev_val])

    # Delete the key at a specified index.
    def delete_key(self, idx: int):
        n_frames = len(self.keyframes)
        if idx == -1:
            idx = n_frames - 1
        if idx in {0.0, n_frames-1}:
            if n_frames > 2:
                self.delete_key_begin_end(idx == n_frames-1)
            else:
                self.keyframes[idx][1] = copy.copy(MotionCycle.default_state)
        else:
            del self.keyframes[idx]

    # Deletes the key either at the beginning or end, provided there are middle frames.
    def delete_key_begin_end(self, end=True):
        if end:
            del self.keyframes[-1]
            max_t = self.keyframes[-1][0]
            for key in self.keyframes:
                key[0] /= max_t                            # notice how time = 0.0 remains there.

        else:
            del self.keyframes[0]
            min_t = self.keyframes[0][0]
            for key in self.keyframes:
                key[0] = (key[0] - min_t) / (1.0 - min_t)  # notice how time = 1.0 remains there.

    def frame_left_of(self, t):
        for i in range(len(self.keyframes)-1):
            if self.keyframes[i][0] <= t <= self.keyframes[i+1][0]:
                return i

    # Evaluate cardinal spline to get exact frame.
    # NOTE: ANGLES MUST ALREADY BE ADJUSTED TO BE NICE FOR INTERPOLATION!!
    def frame(self, u):
        assert 0 <= u <= self.duration and len(self.keyframes) >= 2
        t = u / self.duration
        frames = self.keyframes
        for i in range(0, len(frames)-1):
            if frames[i][0] <= t <= frames[i+1][0]:
                # Remap t to segment range
                t_ = (t - frames[i][0]) / (frames[i+1][0] - frames[i][0])
                # Get bezier points and slopes
                p0 = (2 * frames[i][1]) - frames[i+1][1] if i == 0 else frames[i-1][1]
                p3 = (2 * frames[i+1][1]) - frames[i][1] if i + 2 == len(frames) else frames[i+2][1]
                p1, p2 = frames[i][1], frames[i+1][1]
                m1, m2 = (p2 - p0) * self.tension, (p3 - p1) * self.tension
                # Use Hermite polynomials for interpolation
                total = ((2*np.power(t_, 3)) - (3*np.power(t_, 2)) + 1) * p1
                total += ((np.power(t_, 3)) - (2*np.power(t_, 2)) + t_) * m1
                total += ((-2*np.power(t_, 3)) + (3*np.power(t_, 2))) * p2
                total += (np.power(t_, 3) - np.power(t_, 2)) * m2
                return total

    def __repr__(self):
        #return str([key[0] for key in self.keyframes])
        return self.name


# Use a state vector to update limbs
def update_limbs(state):
    assert len(state) == 14
    global limbs
    # Set angles
    for i, angle in enumerate(state[:12]):
        limbs[i].set_angle(angle)
    # Set root positions
    d = np.array(state[12:])
    limbs[0].set_pos(d)
    limbs[3].set_pos(d)
    limbs[6].set_pos(d)


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


# Create / load motion cycles
file = open('motions.txt', 'r+')
n_motions = int(file.readline())
motions = []
for i in range(n_motions):
    label = file.readline()[len('motion')+1:-1]
    duration = float(file.readline())
    tension = float(file.readline())
    n_frames = int(file.readline())
    keyframes = []
    for j in range(n_frames):
        values = np.array(file.readline().rstrip().rsplit(' ')).astype(float)
        keyframes.append([values[0], np.array(values[1:])])

    motions.append(MotionCycle(label, duration, keyframes, tension))

current_motion = 0

# Guide joint
guide = np.array([0, -limblen * 2])
guide_dragged = False

# IK Params
n_epochs = 30
chain = []
target = None  # in guide space
min_reach, max_reach = 0, 0

# Timeline Params.
# When a frame is selected, the pointer will snap to the time of that frame.
# When the pointer is moved, there will be no selected frame (we'll set it to -1).
timeline_width = window_w * 0.8
offset = ((window_w - timeline_width) / 2)
selected_frame = 0  # -1 = none selected
frame_dragged = False
t = 0


# GUI Params
joint_radius = 7
click_radius = joint_radius * 1.5
running = True

# Switches
main_mode = False  # modes: True = Controller mode, False = Pose mode
fkik_mode = False  # modes: True = IK Mode or False = FK Mode
show_joints = False
onion = True
playing = False


# Main runner
def rerun():
    global w, torso, thigh1, thigh2, joint_radius, skull, n_epochs, chain, target, running, fkik_mode, show_joints,\
           min_reach, max_reach, main_mode, timeline_width, t, motions, current_motion, offset, selected_frame, guide,\
           onion, playing

    # Setup
    w.configure(background='black')

    while playing or running:
        w.delete('all')

        if not main_mode:
            # Update the character joints (NOTE: If chain[0] is None it means we moved "the" root.)
            if len(chain) > 0 and chain[0] is not None and target is not None:
                # Solve IK problem (FK problem is just IK problem of chain length 1)
                for i in range(n_epochs):
                    J = jacobian(chain)
                    J_pinv = np.linalg.pinv(J)
                    current = chain[-1].distal()
                    update = np.array([np.dot(J_pinv, (target - current))]).T
                    theta_n = np.array([limb.angle for limb in chain])
                    theta_nplus1 = np.array([theta_n]).T + update
                    for theta, limb in zip(theta_nplus1.T[0], chain):
                        limb.set_angle(theta)

            if len(chain) > 0 and selected_frame != -1:
                # Update the current keyframe based on limbs
                new_state = motions[current_motion].keyframes[selected_frame][1]
                for i, limb in enumerate(limbs):
                    new_state[i] = limb.angle
                    if limb.parent is None:
                        new_state[-2] = limb.d[0]
                        new_state[-1] = limb.d[1]
                    motions[current_motion].update_state(selected_frame, new_state)

            # Indicate FKIK mode
            w.create_text(window_w/2, 90, text='IK Mode' if fkik_mode else 'FK Mode', fill='white', font='Avenir 20')

            # Indicate Motion Cycle Numbers
            w.create_text(window_w/2, 50, text='Editing: '+str(motions[current_motion]).upper(), fill='orange', font='Avenir 30')

            # Draw timeline
            # 1. Main horizontal line
            selected = selected_frame != -1
            x = offset
            y = window_h - 150
            w.create_line(x, y, x + timeline_width, y, fill='white')
            w.create_line(x, y-40, x, y+40, fill='white')
            w.create_line(x + timeline_width, y-40, x + timeline_width, y+40, fill='white')
            # 2. Frame lines
            for i in range(len(motions[current_motion].keyframes)):
                x = (motions[current_motion].keyframes[i][0] * timeline_width) + offset
                w.create_line(x, y-25, x, y+25, width=10, fill='red' if i == selected_frame else 'blue')

            # 3. Current frame pointer
            x = (t * timeline_width) + offset
            w.create_polygon(x, y+50, x-15, y+50+30, x+15, y+50+30, fill='red' if selected else 'blue')

            # 4. Indicate frame number or t
            frame_text = 'Frame ' + str(selected_frame+1) + '/' + str(len(motions[current_motion].keyframes))
            w.create_text(window_w / 2, window_h - 40, text=frame_text if selected else 't='+str(np.round(t, decimals=2)), fill='red', font='Avenir 30')

            # Draw target
            if target is not None:
                # Draw target circle
                center = A(target + guide)
                w.create_oval(*(center - joint_radius), *(center + joint_radius), fill='', outline='yellow')

                # Draw reach circles
                if chain[0] is not None:
                    origin = torso.proximal() + guide if fkik_mode else chain[0].proximal() + guide
                    w.create_oval(*A(origin - max_reach), *A(origin + max_reach), fill='', outline='blue')
                    if fkik_mode:
                        w.create_oval(*A(origin - min_reach), *A(origin + min_reach), fill='', outline='purple')

            # Draw guide
            apothem = 10
            w.create_rectangle(*A(guide + np.array([-apothem, +apothem])), *A(guide + np.array([+apothem, -apothem])), fill='orange')

            # Draw character + Onion Skins
            loop_min = max(0, selected_frame-2) if onion and selected_frame != -1 else 0
            loop_max = selected_frame+1 if onion and selected_frame != -1 else 1
            # We draw the onions first, and the actual one last.
            for i in range(loop_min, loop_max):
                multiplier = np.power(0.4, loop_max - 1 - i)
                color = _from_rgb(tuple((np.array([255, 255, 255]) * multiplier).astype(int)))

                # Set the limbs to be at the current state, if there are any (else break).
                t_prev = motions[current_motion].keyframes[i][0] if selected_frame != -1 else t
                duration = motions[current_motion].duration
                state = motions[current_motion].frame(t_prev * duration)
                update_limbs(state)

                # 1. Draw fake head
                center = A(guide + ((skull.proximal() + skull.distal()) / 2))
                head_radius = limblen * 0.5
                w.create_oval(*(center - head_radius), *(center + head_radius), fill=color, outline='', width=4)
                # 2. Draw upper body
                root_loc = A(torso.proximal() + guide)
                w.create_oval(*(root_loc - joint_radius), *(root_loc + joint_radius), fill=color) if show_joints else None
                for pair in get_line_points(torso):
                    p0, p1 = A(Ainv(pair[0]) + guide), A(Ainv(pair[1]) + guide)
                    w.create_line(*p0, *p1, fill=color, width=4)
                    w.create_oval(*(p1 - joint_radius), *(p1 + joint_radius), fill=color) if show_joints else None
                # 3. Draw legs
                pairs = get_line_points(thigh1)
                pairs.extend(get_line_points(thigh2))
                for pair in pairs:
                    p0, p1 = A(Ainv(pair[0]) + guide), A(Ainv(pair[1]) + guide)
                    w.create_line(*p0, *p1, fill=color, width=4)
                    w.create_oval(*(p1 - joint_radius), *(p1 + joint_radius), fill=color) if show_joints else None

        else:
            pass

        w.update()
        running = False
        if playing:
            # e.g. After n=100 iterations we'll reach t = 100 * (1 / 100) = 1, but the animation will have taken
            # 100 * (duration / 100) = duration amount of time.
            divisions = 100
            dt = 1.0 / float(divisions)
            t = (t + dt) % 1.0
            time.sleep(dt * motions[current_motion].duration)


# From https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


# Key bind
def key_pressed(event):
    global fkik_mode, main_mode, selected_frame, motions, current_motion, t, running, playing
    n_frames = len(motions[current_motion].keyframes)
    if event.char == 'm':
        main_mode = not main_mode

    if not main_mode:
        if not playing:
            if event.char == ' ':
                fkik_mode = not fkik_mode

            if event.char == 'n':
                motions[current_motion].insert_key(t)
                approx_frame = motions[current_motion].frame_left_of(t)
                selected_frame = approx_frame + 1 if t != 0.0 else approx_frame

            if event.char == 'a':
                approx_frame = motions[current_motion].frame_left_of(t)
                selected_frame = (selected_frame - 1) % n_frames if selected_frame != -1 else approx_frame
                t = motions[current_motion].keyframes[selected_frame][0]
            if event.char == 'd':
                approx_frame = motions[current_motion].frame_left_of(t)
                selected_frame = (selected_frame + 1) % n_frames if selected_frame != -1 else approx_frame + 1
                t = motions[current_motion].keyframes[selected_frame][0]

            if event.char == 'x' and selected_frame != -1:
                motions[current_motion].delete_key(selected_frame)
                approx_frame = motions[current_motion].frame_left_of(t)
                selected_frame = selected_frame % (n_frames - 1) if selected_frame != -1 else approx_frame
                t = motions[current_motion].keyframes[selected_frame][0]

            if event.char == '.':
                current_motion = (current_motion + 1) % len(motions)
                selected_frame = 0
                t = 0

            if event.char == ',':
                current_motion = (current_motion - 1) % len(motions)
                selected_frame = 0
                t = 0

        if event.char == 'p':
            playing = not playing
            selected_frame = -1

    if not running:
        running = True
        rerun()


# MOUSE METHODS ————————————————————————————————————————————
def mouse_click(event):
    '''
         Fires when mouse button is first pressed down.
         Here, we'll build the kinematic chain for the IK problem.
    '''
    global click_radius, limbs, chain, torso, main_mode, selected_frame, motions, current_motion,\
           timeline_width, offset, t, running, guide, guide_dragged, frame_dragged
    clicked_pt = Ainv(np.array([event.x, event.y]))

    if not main_mode and not playing:
        # Move limbs using FKIK
        if selected_frame != -1:
            for limb in limbs:
                # Check only the proximal joint of each limb. Unless it's an end-effector, then check it's distal too.
                is_endeff = len(limb.children) == 0
                b1 = np.linalg.norm(limb.proximal() + guide - clicked_pt) < click_radius
                b2 = is_endeff and np.linalg.norm(limb.distal() + guide - clicked_pt) < click_radius
                if b1 or b2:
                    # Create IK chain
                    if fkik_mode:
                        chain = limb.get_chain_to_root(include_self=b2, max_size=10000)  # TODO: account for variable max chain size
                        chain.reverse()
                    # Or create FK "chain".
                    else:
                        chain = [limb if b2 else limb.parent]  # NOTE: Chain could contain "NONE" in case that a root is selected
                    break

        # Select frames
        dx, dy = 10, 30
        for i in range(len(motions[current_motion].keyframes)):
            x, y = Ainv(np.array([(motions[current_motion].keyframes[i][0] * timeline_width) + offset, window_h - 150]))

            if abs(clicked_pt[0] - x) < dx and abs(clicked_pt[1] - y) < dy:
                selected_frame = i
                t = motions[current_motion].keyframes[i][0]
                frame_dragged = True

        # Select pointer for dragging
        triangle_center = Ainv(np.array([(t * timeline_width) + offset, window_h - 100]))
        if np.linalg.norm(triangle_center - clicked_pt) < 30:
            selected_frame = -1

        # Select guide
        if selected_frame != -1:
            guide_dragged = np.linalg.norm(clicked_pt - guide) < 20

    if not running and not playing:
        running = True
        rerun()


def left_drag(event):
    '''
        Set the target based on click position + resolve FKIK problem, and redraw.
    '''
    global target, chain, running, torso, thigh1, thigh2, min_reach, max_reach, main_mode, selected_frame,\
           t, offset, timeline_width, guide, guide_dragged, frame_dragged
    dragged_pt = Ainv(np.array([event.x, event.y]))
    if not main_mode and not playing:
        # Manipulate Joints
        if len(chain) > 0:
            target = dragged_pt - guide
            if chain[0] is None:  # "The" root is selected (there actually are 3 roots)
                torso.set_pos(target)
                thigh1.set_pos(target)
                thigh2.set_pos(target)
            else:
                # Compute max and min reach
                rootpos = chain[0].proximal()
                min_reach, max_reach = minmax_radii(chain)

                # Compute target (If norm is 0 just keep it where it is.)
                norm = np.linalg.norm(target - rootpos)
                if norm != 0:
                    hat = (target - rootpos) / norm
                    target = rootpos + (max(min_reach, min(max_reach, norm)) * hat)

        # Drag the frame pointer
        if selected_frame == -1:
            t = max(min((A(dragged_pt)[0] - offset), timeline_width), 0) / timeline_width

        # Drag the guide
        if selected_frame != -1 and guide_dragged:
            guide = dragged_pt

        # Drag keyframes around
        if selected_frame != -1 and frame_dragged:
            n_frames = len(motions[current_motion].keyframes)
            epsilon = 10
            max_lim = timeline_width * motions[current_motion].keyframes[(selected_frame + 1) % n_frames][0]
            min_lim = timeline_width * motions[current_motion].keyframes[(selected_frame - 1) % n_frames][0]
            if selected_frame in {n_frames-1, 0}:
                max_lim = 10000
                min_lim = -10000
            t_ = max(min((A(dragged_pt)[0] - offset), max_lim - epsilon), min_lim + epsilon) / timeline_width
            motions[current_motion].update_t(selected_frame, t_)
            if selected_frame not in {0, n_frames-1}:
                t = t_

    if not running and not playing:
        running = True
        rerun()


def mouse_release(event):
    global chain, target, running, main_mode, guide_dragged, frame_dragged
    if not main_mode:
        chain = []
        guide_dragged = False
        frame_dragged = False
        target = None

    if not running and not playing:
        running = True
        rerun()


def save(event):
    global motions
    file = open('motions.txt', 'r+')
    answer = askyesno("Save Motions", 'Save to ' + str(file.name) + '?')
    if answer:
        file.seek(0)
        file.truncate()
        file.write(str(len(motions)) + "\n")
        for motion in motions:
            file.write('Motion ' + motion.name.upper() + "\n")
            file.write(str(float(motion.duration)) + "\n")
            file.write(str(float(motion.tension)) + "\n")
            file.write(str(len(motion.keyframes)) + "\n")
            for frame in motion.keyframes:
                file.write(str(frame[0]) + " ")
                for i, value in enumerate(frame[1]):
                    end = ' ' if i < 13 else '\n'
                    file.write(str(float(value)) + end)


def motion(event):
    pass


# Mouse bind
w.bind('<Motion>', motion)
w.bind("<Button-1>", mouse_click)
w.bind("<ButtonRelease-1>", mouse_release)
w.bind('<B1-Motion>', left_drag)

root.bind("<Command-s>", save)
root.bind("<KeyPress>", key_pressed)
w.pack()

# We call the main function
if __name__ == '__main__':
    rerun()

# Necessary line for Tkinter
mainloop()