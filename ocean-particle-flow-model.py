import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class OceanFlowModel:
    def __init__(self, domain_size, resolution, initial_conditions):
        """
        Initialize the ocean flow and temperature model
        
        Parameters:
        - domain_size: Tuple representing spatial dimensions (x, y, z)
        - resolution: Spatial resolution of the grid
        - initial_conditions: Dictionary of initial temperature and flow conditions
        """
        self.domain_size = domain_size
        self.resolution = resolution
        self.initial_conditions = initial_conditions
        
        # Fundamental oceanographic parameters
        self.params = {
            'density': 1025,  # kg/m³ (typical seawater density)
            'specific_heat_capacity': 4186,  # J/(kg·K) 
            'thermal_diffusivity': 1.4e-7,  # m²/s
            'viscosity': 1e-6,  # m²/s
        }
    
    def navier_stokes_temperature(self, state, t, external_forces):
        """
        Coupled Navier-Stokes and heat transfer equations
        
        Parameters:
        - state: Current state vector [velocity_x, velocity_y, velocity_z, temperature]
        - t: Time
        - external_forces: Environmental forces (wind, current, etc.)
        
        Returns:
        Rate of change for velocity and temperature
        """
        # Velocity components
        u, v, w, T = state
        
        # Momentum equations (simplified 3D Navier-Stokes)
        du_dt = -(u * np.gradient(u) + v * np.gradient(v) + w * np.gradient(w)) \
                + self.params['viscosity'] * np.laplacian(u) \
                + external_forces['momentum_x']
        
        dv_dt = -(u * np.gradient(u) + v * np.gradient(v) + w * np.gradient(w)) \
                + self.params['viscosity'] * np.laplacian(v) \
                + external_forces['momentum_y']
        
        dw_dt = -(u * np.gradient(u) + v * np.gradient(v) + w * np.gradient(w)) \
                + self.params['viscosity'] * np.laplacian(w) \
                + external_forces['momentum_z']
        
        # Temperature evolution (advection-diffusion equation)
        dT_dt = -(u * np.gradient(T) + v * np.gradient(T) + w * np.gradient(T)) \
                + self.params['thermal_diffusivity'] * np.laplacian(T) \
                + external_forces['heating']
        
        return [du_dt, dv_dt, dw_dt, dT_dt]
    
    def simulate_ocean_dynamics(self, duration, time_steps):
        """
        Simulate ocean particle flow and temperature dynamics
        
        Parameters:
        - duration: Total simulation time
        - time_steps: Number of time steps
        
        Returns:
        Simulation results for particle trajectories and temperature
        """
        # Define time array
        t = np.linspace(0, duration, time_steps)
        
        # Initial state vector
        initial_state = [
            self.initial_conditions['velocity_x'],
            self.initial_conditions['velocity_y'],
            self.initial_conditions['velocity_z'],
            self.initial_conditions['temperature']
        ]
        
        # External environmental forces
        external_forces = {
            'momentum_x': 0,  # Wind/current influences
            'momentum_y': 0,
            'momentum_z': 0,
            'heating': 0  # Solar radiation, atmospheric conditions
        }
        
        # Solve coupled differential equations
        solution = odeint(
            self.navier_stokes_temperature, 
            initial_state, 
            t, 
            args=(external_forces,)
        )
        
        return solution, t
    
    def visualize_results(self, solution, time_array):
        """
        Visualize simulation results
        
        Parameters:
        - solution: Simulation results
        - time_array: Corresponding time steps
        """
        plt.figure(figsize=(15, 10))
        
        # Velocity components
        plt.subplot(2, 2, 1)
        plt.title('Velocity X Component')
        plt.plot(time_array, solution[:, 0])
        
        plt.subplot(2, 2, 2)
        plt.title('Velocity Y Component')
        plt.plot(time_array, solution[:, 1])
        
        # Temperature variation
        plt.subplot(2, 2, 3)
        plt.title('Temperature Variation')
        plt.plot(time_array, solution[:, 3])
        
        plt.tight_layout()
        plt.show()

# Example usage
initial_conditions = {
    'velocity_x': 0.5,  # m/s
    'velocity_y': 0.3,  # m/s
    'velocity_z': 0.1,  # m/s
    'temperature': 28.5  # °C (typical Red Sea surface temperature)
}

model = OceanFlowModel(
    domain_size=(500, 300, 200),  # km dimensions of Red Sea/Gulf of Aden
    resolution=10,  # 10 km grid resolution
    initial_conditions=initial_conditions
)

# Run simulation
solution, time_array = model.simulate_ocean_dynamics(
    duration=30,  # days
    time_steps=1000
)

# Visualize results
model.visualize_results(solution, time_array)
