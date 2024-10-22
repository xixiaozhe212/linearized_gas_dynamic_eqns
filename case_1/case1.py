import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 80  # length of domain
N = 500
dx = L / N
dt = 0.1 * dx
t_max = 60
t_plot_interval = 6

# Initialize x
x = np.linspace(-L / 2, L / 2, N)

# Initial conditions
rho = np.where(np.abs(x) < 2, 1.1, 1)
u = np.zeros(N)

# Arrays for storing updated values
rho_next = np.zeros(N)
u_next = np.zeros(N)
rho_exact = np.copy(rho)
u_exact = np.copy(u)

def update_values(rho, u):
    for i in range(1, N - 1):
        FLrho = 0.5 * u[i - 1] + 0.5 * u[i] - 0.5 * rho[i] + 0.5 * rho[i - 1]
        FLu = 0.5 * rho[i - 1] + 0.5 * rho[i] - 0.5 * u[i] + 0.5 * u[i - 1]

        FRrho = 0.5 * u[i] + 0.5 * u[i + 1] - 0.5 * rho[i + 1] + 0.5 * rho[i]
        FRu = 0.5 * rho[i] + 0.5 * rho[i + 1] - 0.5 * u[i + 1] + 0.5 * u[i]

        rho_next[i] = rho[i] + dt / dx * (FLrho - FRrho)
        u_next[i] = u[i] + dt / dx * (FLu - FRu)
    return rho_next, u_next

# Time-stepping loop
for t in np.arange(0, t_max, dt):
    if t % t_plot_interval == 0:
        plt.figure(figsize=(10, 8))

        # Plot rho and rho_exact
        plt.subplot(2, 1, 1)
        plt.plot(x, rho, label=f'rho at t={t:.1f}')
        plt.plot(x, rho_exact, label=f'rho_exact at t={t:.1f}')
        plt.title("Density (rho) and rho_exact")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.ylim(0.98, 1.12)

        # Plot u and u_exact
        plt.subplot(2, 1, 2)
        plt.plot(x, u, label=f'u at t={t:.1f}')
        plt.plot(x, u_exact, label=f'u_exact at t={t:.1f}')
        plt.title("Velocity (u) and u_exact")
        plt.xlabel("x")
        plt.ylabel("Velocity")
        plt.legend()
        plt.ylim(-0.06, 0.06)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'plot_t_{t:.1f}.png')
        plt.close()  # Close the figure to free up memory

    # Apply reflective boundary conditions
    rho[0] = rho[1]
    u[0] = -u[1]
    rho[-1] = rho[-2]
    u[-1] = -u[-2]

    rho_next, u_next = update_values(rho, u)
    rho, u = rho_next.copy(), u_next.copy()
    rho_next[0] = rho_next[1]
    u_next[0] = -u_next[1]
    rho_next[-1] = rho_next[-2]
    u_next[-1] = -u_next[-2]

    # Update rho_exact and u_exact based on shifted initial conditions
    for i in range(1, N - 1):
        x_shifted1 = x[i] + t
        x_shifted2 = x[i] - t

        rho01 = 1.1 if np.abs(x_shifted1) < 2 else 1
        u01 = 0
        rho02 = 1.1 if np.abs(x_shifted2) < 2 else 1
        u02 = 0

        rho_exact[i] = 0.5 * (rho01 + rho02) - 0.5 * (u01 + u02)
        u_exact[i] = 0.5 * (rho02 - rho01) + 0.5 * (u01 + u02)

# Final plot settings
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(x, rho, label='Final rho')
plt.plot(x, rho_exact, label='Final rho_exact')
plt.title("Final Density (rho) and rho_exact")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.ylim(0.98, 1.12)

plt.subplot(2, 1, 2)
plt.plot(x, u, label='Final u')
plt.plot(x, u_exact, label='Final u_exact')
plt.title("Final Velocity (u) and u_exact")
plt.xlabel("x")
plt.ylabel("Velocity")
plt.legend()
plt.ylim(-0.06, 0.06)

plt.tight_layout()
plt.show()