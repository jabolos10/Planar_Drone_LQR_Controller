"""
Controls:
    Click anywhere  — set new goal position
    R               — reset drone to start
    D               — apply random disturbance
    SPACE           — pause / resume
    ESC             — quit
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import pygame
import sys
import math

# -- Physical parameters --------------------------
m    = 1.0
g    = 9.81
I    = 0.05
L    = 0.20
F0   = m * g / 2.0
FMAX = 15.0
FMIN = 0.0

# --Linearised A, B matrices -------------------
A = np.array([
    [0, 1,  0, 0,   0,  0],
    [0, 0,  0, 0,  -g,  0],
    [0, 0,  0, 1,   0,  0],
    [0, 0,  0, 0,   0,  0],
    [0, 0,  0, 0,   0,  1],
    [0, 0,  0, 0,   0,  0],
])
B = np.array([
    [0,    0   ],
    [0,    0   ],
    [0,    0   ],
    [1/m,  1/m ],
    [0,    0   ],
    [L/I, -L/I ],
])

#-- LQR gain matrix ---------------------------------
Q = np.diag([20.0, 2.0, 20.0, 2.0, 15.0, 1.0])
R = np.diag([0.5, 0.5])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("K computed. Eigenvalues of A-BK:")
for e in np.linalg.eigvals(A - B @ K):
    print(f"  {e.real:.3f} + {e.imag:.3f}j")

# --Nonlinear dynamics ------------------------------
def dynamics(state, u):
    x, xd, y, yd, th, thd = state
    Ft  = u[0] + u[1]
    tau = L * (u[0] - u[1])
    return np.array([xd,
                     -Ft * math.sin(th) / m,
                     yd,
                      Ft * math.cos(th) / m - g,
                     thd,
                     tau / I])

# -- LQR control ---------------------------
def lqr_control(state, goal):
    ref   = np.array([goal[0], 0, goal[1], 0, 0, 0])
    err   = state - ref
    du    = K @ err
    F1    = np.clip(F0 - du[0], FMIN, FMAX)
    F2    = np.clip(F0 - du[1], FMIN, FMAX)
    return np.array([F1, F2])

# -- RK4 ----------------------------------
def rk4(state, u, dt):
    k1 = dynamics(state,           u)
    k2 = dynamics(state + dt/2*k1, u)
    k3 = dynamics(state + dt/2*k2, u)
    k4 = dynamics(state + dt  *k3, u)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# -- PyGame setup --------------------------
W, H   = 900, 650
FPS    = 60
DT     = 1 / FPS
SCALE  = 90           # pixels per metre
ORIGIN = (W//2, H//2) # world (0,0) maps here

# colours
BG       = (15,  18,  30)
GRID     = (30,  36,  54)
BODY     = (96,  165, 250)   # blue
ARM      = (71,  122, 186)
PROP_L   = (167, 139, 250)   # left rotor -> purple
PROP_R   = (52,  211, 153)   # right rotor -> green
THRUST_C = (251, 191,  36)   # thrust flame -> blue kind of
GOAL_C   = (251, 191,  36)
TRAIL_C  = (60,  80,  120)
TEXT_C   = (200, 210, 230)
PANEL_BG = (22,  28,  46)

def world_to_px(wx, wy):

    px = int(ORIGIN[0] + wx * SCALE)
    py = int(ORIGIN[1] - wy * SCALE)
    return px, py

def draw_drone(surf, state, u):
    x, _, y, _, th, _ = state
    cx, cy = world_to_px(x, y)
    arm_px = int(L * SCALE * 1.6)   # visual arm length

    # arm
    dx = int(math.cos(th) * arm_px)
    dy = int(math.sin(th) * arm_px)
    pygame.draw.line(surf, ARM, (cx-dx, cy+dy), (cx+dx, cy-dy), 4)


    # thrust flames -> length proportional to thrust
    for sign, F, col in [(-1, u[0], PROP_L), (1, u[1], PROP_R)]:
        bx = cx + sign * dx
        by = cy - sign * dy
        flame_len = int((F / FMAX) * 22)


        # perpendicular direction -> downward from rotor
        perp_x = int( math.sin(th) * flame_len)
        perp_y = int( math.cos(th) * flame_len)
        if flame_len > 2:
            pygame.draw.line(surf, THRUST_C,
                             (bx, by),
                             (bx + perp_x, by + perp_y), 3)


    # rotor discs
    for sign, col in [(-1, PROP_L), (1, PROP_R)]:
        bx = cx + sign * dx
        by = cy - sign * dy
        pygame.draw.circle(surf, col, (bx, by), 8)
        pygame.draw.circle(surf, (255,255,255), (bx, by), 8, 1)

    # body centre
    pygame.draw.circle(surf, BODY, (cx, cy), 10)
    pygame.draw.circle(surf, (255,255,255), (cx, cy), 10, 1)


def draw_grid(surf):
    for mx in range(-5, 6):
        px, _ = world_to_px(mx, 0)
        pygame.draw.line(surf, GRID, (px, 0), (px, H), 1)
    for my in range(-4, 5):
        _, py = world_to_px(0, my)
        pygame.draw.line(surf, GRID, (0, py), (W, py), 1)
    # axes
    ox, oy = world_to_px(0, 0)
    pygame.draw.line(surf, (50, 60, 90), (0, oy), (W, oy), 1)
    pygame.draw.line(surf, (50, 60, 90), (ox, 0), (ox, H), 1)



def draw_goal(surf, goal):
    gx, gy = world_to_px(goal[0], goal[1])
    pygame.draw.circle(surf, GOAL_C, (gx, gy), 10, 2)
    pygame.draw.line(surf, GOAL_C, (gx-12, gy), (gx+12, gy), 1)
    pygame.draw.line(surf, GOAL_C, (gx, gy-12), (gx, gy+12), 1)


def draw_trail(surf, trail):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        alpha = int(180 * i / len(trail))
        col = (TRAIL_C[0], TRAIL_C[1], min(255, TRAIL_C[2] + alpha))
        pygame.draw.line(surf, col, trail[i-1], trail[i], 1)



def draw_panel(surf, font_big, font_sm, state, u, goal, sim_t, paused):
    panel_w = 220
    pygame.draw.rect(surf, PANEL_BG, (0, 0, panel_w, H))
    pygame.draw.line(surf, GRID, (panel_w, 0), (panel_w, H), 1)

    x, xd, y, yd, th, thd = state

    lines = [
        ("STATE",        None,        True),
        (f"x     {x:+.3f} m",   TEXT_C,  False),
        (f"y     {y:+.3f} m",   TEXT_C,  False),
        (f"th    {math.degrees(th):+.2f} °", TEXT_C, False),
        (f"ẋ     {xd:+.3f} m/s", TEXT_C, False),
        (f"ẏ     {yd:+.3f} m/s", TEXT_C, False),
        ("",             None,        False),
        ("GOAL",         None,        True),
        (f"x_ref  {goal[0]:+.2f} m", TEXT_C, False),
        (f"y_ref  {goal[1]:+.2f} m", TEXT_C, False),
        ("",             None,        False),
        ("ROTORS",       None,        True),
        (f"F₁    {u[0]:.3f} N",  PROP_L,  False),
        (f"F₂    {u[1]:.3f} N",  PROP_R,  False),
        (f"F₀    {F0:.3f} N",    (120,130,150), False),
        ("",             None,        False),
        ("SIM",          None,        True),
        (f"t     {sim_t:.1f} s", TEXT_C, False),
        (f"dt    {DT*1000:.1f} ms",  TEXT_C, False),
    ]

    y_off = 18
    for text, col, header in lines:
        if text == "":
            y_off += 8
            continue
        if header:
            surf_txt = font_big.render(text, True, (100, 130, 200))
            surf.blit(surf_txt, (14, y_off))
            pygame.draw.line(surf, (40, 50, 80), (14, y_off+18), (panel_w-14, y_off+18), 1)
            y_off += 26
        else:
            surf_txt = font_sm.render(text, True, col or TEXT_C)
            surf.blit(surf_txt, (18, y_off))
            y_off += 20


    hints = [
        "CONTROLS",
        "Click  → set goal",
        "R      → reset",
        "D      → disturbance",
        "SPACE  → pause",
        "ESC    → quit",
    ]
    y_off = H - len(hints)*18 - 14
    for i, h in enumerate(hints):
        col = (100, 130, 200) if i == 0 else (80, 90, 110)
        surf.blit(font_sm.render(h, True, col), (14, y_off + i*18))

    if paused:
        txt = font_big.render("PAUSED", True, (251, 191, 36))
        surf.blit(txt, (panel_w//2 - txt.get_width()//2, H//2 - 10))


# -- Main -------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Planar Drone — LQR Controller")
    clock  = pygame.time.Clock()
    font_b = pygame.font.SysFont("monospace", 13, bold=True)
    font_s = pygame.font.SysFont("monospace", 12)

    INIT_STATE = np.array([2.0, 0.0, 1.0, 0.0, 0.3, 0.0])
    state = INIT_STATE.copy()
    goal  = [0.0, 0.0]
    sim_t = 0.0
    paused = False
    trail  = []
    MAX_TRAIL = 400

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = INIT_STATE.copy()
                    sim_t = 0.0
                    trail.clear()
                elif event.key == pygame.K_d:
                    state[1] += np.random.uniform(-4, 4)   # x velocity kick
                    state[5] += np.random.uniform(-5, 5)   # angular rate kick
                elif event.key == pygame.K_SPACE:
                    paused = not paused

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx > 220:   # outside panel
                    # converting  px → world
                    wx = (mx - ORIGIN[0]) / SCALE
                    wy = (ORIGIN[1] - my) / SCALE
                    goal = [wx, wy]

        if not paused:
            u = lqr_control(state, goal)

            # physics substeps (4× per frame for better accuracy)
            for _ in range(4):
                state = rk4(state, lqr_control(state, goal), DT/4)

            sim_t += DT

            # trail
            px, py = world_to_px(state[0], state[2])
            trail.append((px, py))
            if len(trail) > MAX_TRAIL:
                trail.pop(0)

        # -- Drawinggg ----------------------------------------------------
        screen.fill(BG)
        draw_grid(screen)
        draw_trail(screen, trail)
        draw_goal(screen, goal)

        u_draw = lqr_control(state, goal)
        draw_drone(screen, state, u_draw)

        draw_panel(screen, font_b, font_s, state, u_draw, goal, sim_t, paused)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
