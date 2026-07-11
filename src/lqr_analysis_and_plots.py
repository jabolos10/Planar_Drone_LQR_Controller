

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# -- Physical parameters ---------------------------------------------------
m   = 1.0
g   = 9.81
I   = 0.05   # moment of inertia about z-axis
L   = 0.20   # half-arm length

F_max = 15.0  # maximum rotor thrust
F_min = 0.0   # minimum rotor thrust
F0    = m * g / 2.0  # hover thrust per rotor

# --- Linearised state-space matrices ----------------------------------------------

A = np.array([
    [0, 1,    0, 0,     0,    0],
    [0, 0,    0, 0,   -g,    0],
    [0, 0,    0, 1,     0,    0],
    [0, 0,    0, 0,     0,    0],
    [0, 0,    0, 0,     0,    1],
    [0, 0,    0, 0,     0,    0],
])

B = np.array([
    [0,       0     ],
    [0,       0     ],
    [0,       0     ],
    [1/m,     1/m   ],
    [0,       0     ],
    [L/I,    -L/I   ],
])


# ── LQR cost matrices -------------------------------------------
Q = np.diag([20.0, 2.0, 20.0, 2.0, 15.0, 1.0])
R = np.diag([0.5, 0.5])



# --- Solving continuous-time algebraic Riccati equation= CARE:
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P  # LQR gain matrix with Ks

print("LQR gain matrix K:")
print(np.array2string(K, precision=4, suppress_small=True))
print(f"\nHover thrust per rotor: F0 = {F0:.3f} N")
print(f"Eigenvalues of (A - B@K):")
eigs = np.linalg.eigvals(A - B @ K)
for e in eigs:
    print(f"  {e:.4f}")


# --- Nonlinear dynamics -------------------------------------------------
def nonlinear_dynamics(state, u):
    """state derivative for the nonlinear model"""
    x, x_dot, y, y_dot, th, th_dot = state
    F1, F2 = u
    Ftot = F1 + F2
    tau  = L * (F1 - F2)
    return np.array([
        x_dot,
        -Ftot * np.sin(th) / m,
        y_dot,
         Ftot * np.cos(th) / m - g,
        th_dot,
        tau / I,
    ])


#----- LQR control law -------------------------------------------
def lqr_control(state, goal):
    """
    Compute rotor thrusts with LQR
    error = state - goal_state
    u = u_nominal - K * error
    """
    goal_state = np.array([goal[0], 0, goal[1], 0, 0, 0])
    error = state - goal_state
    delta_u = K @ error
    F1 = F0 - delta_u[0]
    F2 = F0 - delta_u[1]
    F1 = np.clip(F1, F_min, F_max)
    F2 = np.clip(F2, F_min, F_max)
    return np.array([F1, F2])



# --- RK4 integrator -------------------------------------------
def rk4_step(state, u, dt):
    k1 = nonlinear_dynamics(state,        u)
    k2 = nonlinear_dynamics(state + dt/2*k1, u)
    k3 = nonlinear_dynamics(state + dt/2*k2, u)
    k4 = nonlinear_dynamics(state + dt*k3,   u)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)






# --Simulation-------------------------------------------------------
def simulate(initial_state, goal, t_end=10.0, dt=0.005):

    state = np.array(initial_state, dtype=float)
    t = 0.0
    history = {'t':[], 'x':[], 'y':[], 'th':[], 'x_dot':[], 'y_dot':[], 'th_dot':[], 'F1':[], 'F2':[]}

    while t <= t_end:
        u = lqr_control(state, goal)
        for key, val in zip(['t','x','y','th','x_dot','y_dot','th_dot','F1','F2'],
                             [t, state[0], state[2], state[4],
                              state[1], state[3], state[5], u[0], u[1]]):
            history[key].append(val)
        state = rk4_step(state, u, dt)
        t += dt

    return {k: np.array(v) for k, v in history.items()}



# -- Scenario 1: hover stabilisation from offset position -----------------------
print("\n--- Scenario 1: Stabilisation from x=2m, y=1m offset ")
h1 = simulate(initial_state=[2.0, 0, 1.0, 0, 0.3, 0], goal=[0.0, 0.0], t_end=8.0)


# -- Scenario 2: step in goal position (trajectory tracking) --------------------
print("--- Scenario 2: Step from hover to goal (3m, 2m) ")
h2 = simulate(initial_state=[0.0, 0, 0.0, 0, 0.0, 0], goal=[3.0, 2.0], t_end=8.0)


# -- Scenario 3: disturbance rejection ------------------------------------------
print("--- Scenario 3: Disturbance rejection (velocity kick at t=2s) ")
state0 = np.array([0.0, 0, 0.0, 0, 0.0, 0])
goal3  = [0.0, 0.0]
dt     = 0.005
t_kick = 2.0
states, times, u_hist = [state0.copy()], [0.0], []
t = 0.0
while t < 6.0:
    if abs(t - t_kick) < dt/2:  # applying impulse
        state0[1] += 3.0   # x velocity kick
        state0[5] += 4.0   # angular rate kick
    u = lqr_control(state0, goal3)
    u_hist.append(u)
    state0 = rk4_step(state0, u, dt)
    t += dt
    states.append(state0.copy()); times.append(t)
states = np.array(states); times = np.array(times)
u_hist = np.array(u_hist)




# -- Plots ----------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
fig.suptitle("Planar Drone — LQR Controller: Simulation Results", fontsize=14, fontweight='bold')

colors = {'x':'#2563eb', 'y':'#16a34a', 'th':'#dc2626', 'F1':'#7c3aed', 'F2':'#0891b2'}


# --- Row 1: Stabilisation ---
ax = axes[0,0]
ax.plot(h1['t'], h1['x'],  color=colors['x'],  label='x [m]')
ax.plot(h1['t'], h1['y'],  color=colors['y'],  label='y [m]')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_title("Sc. 1 — Position (stabilisation)"); ax.set_xlabel("t [s]"); ax.set_ylabel("m")
ax.legend(); ax.grid(True, alpha=0.3)


ax = axes[0,1]
ax.plot(h1['t'], np.degrees(h1['th']), color=colors['th'])
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_title("Sc. 1 — Attitude θ"); ax.set_xlabel("t [s]"); ax.set_ylabel("deg")
ax.grid(True, alpha=0.3)

ax = axes[0,2]
ax.plot(h1['t'], h1['F1'], color=colors['F1'], label='F₁')
ax.plot(h1['t'], h1['F2'], color=colors['F2'], label='F₂', ls='--')
ax.axhline(F0, color='gray', ls=':', lw=0.8, label='F₀=mg/2')
ax.set_title("Sc. 1 — Rotor thrusts"); ax.set_xlabel("t [s]"); ax.set_ylabel("N")
ax.legend(); ax.grid(True, alpha=0.3)



# --- Row 2: Trajectory tracking ---
ax = axes[1,0]
ax.plot(h2['t'], h2['x'],  color=colors['x'],  label='x [m]')
ax.plot(h2['t'], h2['y'],  color=colors['y'],  label='y [m]')
ax.axhline(3.0, color=colors['x'], ls=':', lw=0.8, label='x_ref=3m')
ax.axhline(2.0, color=colors['y'], ls=':', lw=0.8, label='y_ref=2m')
ax.set_title("Sc. 2 — Position tracking"); ax.set_xlabel("t [s]"); ax.set_ylabel("m")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.plot(h2['t'], np.degrees(h2['th']), color=colors['th'])
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_title("Sc. 2 — Attitude θ"); ax.set_xlabel("t [s]"); ax.set_ylabel("deg")
ax.grid(True, alpha=0.3)

ax = axes[1,2]
ax.plot(h2['t'], h2['F1'], color=colors['F1'], label='F₁')
ax.plot(h2['t'], h2['F2'], color=colors['F2'], label='F₂', ls='--')
ax.axhline(F0, color='gray', ls=':', lw=0.8, label='F₀')
ax.set_title("Sc. 2 — Rotor thrusts"); ax.set_xlabel("t [s]"); ax.set_ylabel("N")
ax.legend(); ax.grid(True, alpha=0.3)



# --- Row 3: Disturbance rejection ---
ax = axes[2,0]
ax.plot(times, states[:,0], color=colors['x'], label='x [m]')
ax.plot(times, states[:,2], color=colors['y'], label='y [m]')
ax.axvline(t_kick, color='red', ls=':', lw=1.2, label='disturbance')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_title("Sc. 3 — Disturbance rejection"); ax.set_xlabel("t [s]"); ax.set_ylabel("m")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2,1]
ax.plot(times, np.degrees(states[:,4]), color=colors['th'])
ax.axvline(t_kick, color='red', ls=':', lw=1.2)
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_title("Sc. 3 — Attitude θ"); ax.set_xlabel("t [s]"); ax.set_ylabel("deg")
ax.grid(True, alpha=0.3)

ax = axes[2,2]
ax.plot(times[:-1], u_hist[:,0], color=colors['F1'], label='F₁')
ax.plot(times[:-1], u_hist[:,1], color=colors['F2'], label='F₂', ls='--')
ax.axvline(t_kick, color='red', ls=':', lw=1.2, label='disturbance')
ax.axhline(F0, color='gray', ls=':', lw=0.8, label='F₀')
ax.set_title("Sc. 3 — Rotor thrusts"); ax.set_xlabel("t [s]"); ax.set_ylabel("N")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("drone_results.png", dpi=150, bbox_inches='tight')
plt.close()




# -- 2D trajectory plot of x and y -------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Planar Drone — Flight Trajectories (XY plane)", fontsize=13, fontweight='bold')

ax = axes2[0]
ax.plot(h1['x'], h1['y'], color='#2563eb', lw=1.5, label='trajectory')
ax.plot(h1['x'][0], h1['y'][0], 'go', ms=8, label='start (2, 1)')
ax.plot(0, 0, 'r*', ms=12, label='goal (0, 0)')
ax.set_title("Sc. 1 — Stabilisation"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'box')

ax = axes2[1]
ax.plot(h2['x'], h2['y'], color='#16a34a', lw=1.5, label='trajectory')
ax.plot(h2['x'][0], h2['y'][0], 'go', ms=8, label='start (0, 0)')
ax.plot(3, 2, 'r*', ms=12, label='goal (3, 2)')
ax.set_title("Sc. 2 — Tracking"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'box')

plt.tight_layout()
plt.savefig("drone_trajectories.png", dpi=150, bbox_inches='tight')
plt.close()
