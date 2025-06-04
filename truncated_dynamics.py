import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ------------------------------
# Helper functions for nonlinear terms
# ------------------------------
#def d_s_arr(rho):
#    
#    alpha = np.pi / 2 - 1
#    term1 = 1 - alpha * rho
#    term2 = (alpha * (2 * alpha - 1)) / (2 * alpha + 1) * rho**2
#    return (1 - rho) * (term1 + term2)

def d_s_arr(rho):
    """Calculates the d_s array based on density rho."""
    alpha = np.pi / 2 - 1
    return (1 - rho) * (1 - alpha * rho + ((alpha * (2 * alpha - 1)) / (2 * alpha + 1)) * rho**2)

def D(rho):
    """Computes D(rho) for nonlinear diffusion."""
    return (1 - d_s_arr(rho)) / rho

def s_arr(rho):
    """Computes s(rho) based on density rho."""
    return D(rho) - 1


# ------------------------------
# Main Simulation Class
# ------------------------------
class TruncatedSimulation:
    def __init__(self, v_0=3.0, Lx=10.0, Nx=200, N_harmonics=2, epsilon=0.0, d=0.5, ansatz=None, **inits):
        # Input validation
        assert Lx > 0, "Domain length (Lx) must be positive."
        assert Nx > 0, "Number of spatial points (Nx) must be positive."
        assert N_harmonics > 0, "Number of harmonics must be positive."

        # Domain and parameters
        self.dx = Lx / Nx
        self.Nx = Nx
        self.x = np.linspace(0, Lx, Nx)
        self.v_0 = v_0
        self.Lx = Lx
        self.t = 0.0

        # Harmonic parameters
        self.N_harmonics = N_harmonics
        self.f_tilde = np.zeros((N_harmonics, Nx))
        self.history_f_tilde = []
        if ansatz==None:
            def zero_clousure(arr):
                return 0.0
            self.ansatz=zero_clousure
        else:
            self.ansatz = ansatz

        # Initialize potential V and its gradient
        self.grad_V = self._initialize_potential(epsilon, d)

    def _initialize_potential(self, epsilon, d):
        """Generates a centered hump potential and computes its gradient."""
        V = np.where(
            np.abs(self.x - self.Lx / 2) <= d,
            (1 + np.cos(np.pi / d * (self.x - self.Lx / 2))) * epsilon,
            0.0)

        return np.gradient(V, self.dx)

    # ------------------------------
    # Initialization Methods
    # ------------------------------
    def set_random(self, mean_density=0.5, delta=0.01):
        """Random initialization of density with perturbations."""
        N_theta = 30
        # Random distribution
        f_init = (1 / (2 * np.pi)) * mean_density * np.ones((N_theta, self.Nx)) 
        f_init += (1 / (2 * np.pi)) * delta * np.random.rand(N_theta, self.Nx)

        # Fourier transform initialization
        for n in range(self.N_harmonics):
            angles = np.arange(0, N_theta) * 2 * np.pi / N_theta
            self.f_tilde[n, :] = np.matmul(np.cos(n * angles), f_init) * 2 * np.pi / N_theta

    def set_custom(self, f_init):
        """Custom initialization of harmonics."""
        assert f_init.shape[1] == self.Nx, "Initialization shape mismatch!"
        N_theta = f_init.shape[0]

        for n in range(self.N_harmonics):
            angles = np.arange(0, N_theta) * 2 * np.pi / N_theta
            self.f_tilde[n, :] = np.matmul(np.cos(n * angles), f_init) * 2 * np.pi / N_theta

    # ------------------------------
    # Evolution Function
    # ------------------------------
    def evolve(self, t_f, dt=5e-4, n_record=100):
        """Time evolution of the simulation."""
        # Time-step properties
        self.dt = dt
        self.Nt = int(t_f / dt)

        # Fourier wavevectors
        k = 2 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        k2 = k**2
        dealiasing_mask = np.abs(k) < (2 / 3) * (np.pi / self.dx)

        # Time evolution loop
        for t_step in tqdm(range(self.Nt)):
            # Fourier transform
            f_tilde_hat = np.fft.fft(self.f_tilde, axis=1)
            f_tilde_hat[:, ~dealiasing_mask] = 0  # Apply dealiasing

            # Compute nonlinear terms
            rho = self.f_tilde[0, :]
            P = self.f_tilde[1, :]
            d_s = d_s_arr(rho)
            D_rho = D(rho)
            s_rho = D_rho - 1.0
            grad = np.fft.ifft(1j * k * f_tilde_hat, axis=1).real

            # Update rho with implicit diffusion
            nonlinear_rho = 1j * k * np.fft.fft((1 - rho) * (rho * self.grad_V - self.v_0 * self.f_tilde[1, :]))
            f_tilde_hat[0, :] = (f_tilde_hat[0, :] + self.dt * nonlinear_rho) / (1 + k2 * self.dt)
            f_tilde_hat[0, ~dealiasing_mask] = 0
            self.f_tilde[0, :] = np.fft.ifft(f_tilde_hat[0, :]).real

            # Update higher harmonics
            for n in range(1, self.N_harmonics):
                diffusion = d_s * grad[n, :]
                exclusion = self.f_tilde[n, :] * D_rho * grad[0, :]
                if n == self.N_harmonics-1: #closure with arbitrary ansatz
                    activity = - self.v_0 * (s_rho * self.f_tilde[1, :]*self.f_tilde[n, :] + d_s/2 * (self.f_tilde[n-1, :] + self.ansatz(self.f_tilde)))
                else:
                    activity = - self.v_0 * (s_rho * self.f_tilde[1, :]*self.f_tilde[n, :] + d_s/2 * (self.f_tilde[n-1, :] + self.f_tilde[n+1, :]))
                potential = self.f_tilde[n, :] * (1 - rho) * self.grad_V

                nonlinear_term = diffusion + exclusion + activity + potential
                nonlinear_fourier = 1j * k * np.fft.fft(nonlinear_term)
                f_tilde_hat[n, :] = (f_tilde_hat[n, :] + self.dt * nonlinear_fourier) / (1 + (n**2) * self.dt)
                f_tilde_hat[n, ~dealiasing_mask] = 0
                self.f_tilde[n, :] = np.fft.ifft(f_tilde_hat[n, :]).real

            # Record history
            if t_step % (self.Nt // n_record) == 0:
                self.history_f_tilde.append(self.f_tilde.copy())
            self.t += dt

        self.history_f_tilde = np.array(self.history_f_tilde)

    # ------------------------------
    # Visualization
    # ------------------------------
    def plot_history(self):
        """Plots the evolution of rho over time."""
        history_rho = self.history_f_tilde[:, 0, :]
        plt.imshow(history_rho, aspect='auto', origin='lower', extent=[0, self.Lx, 0, self.t], cmap='coolwarm', vmin=0, vmax=1)
        plt.title(f'Dynamics @ v_0={self.v_0}, t_f={round(self.t, 3)}')
        plt.colorbar(label='Density')
        plt.xlabel('x-coordinate')
        plt.ylabel('Time')
        plt.show()