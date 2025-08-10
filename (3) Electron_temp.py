from math import sin, cos, pi, sqrt, atan, atan2, exp
import numpy as np
import numpy.random as random

def rand_dist(f, a, b, maxf):
    # Returns random numbers between a and b distributed with f
    # Warning: this method is very ineffective for strongly varying f
    while True:
        rx, ry = random.rand(2)
        x_val = a + (b - a) * rx
        y_val = ry * maxf
        if y_val <= f(x_val):
            return x_val

class Electron:
    # Electron mass in MeV
    m = 0.511  
    # Characteristic radius 
    r = 0.0005 

    def __init__(self, x, y, z, dphi, dtheta, T):
        # Takes the initial position and direction (dphi, dtheta) and kinetic energy T in MeV
        self.x = x
        self.y = y
        self.z = z
        self.dphi = dphi
        self.dtheta = dtheta
        self.T = T

    def __repr__(self):
        return f"{self.x} {self.y} {self.z}"

    def propagate(self, s):
        # Propagates the electron by s in the direction specified by dphi, dtheta (this is correct!)
        self.x += s * sin(self.dtheta) * cos(self.dphi)
        self.y += s * sin(self.dtheta) * sin(self.dphi)
        self.z += s * cos(self.dtheta)
        self.T -= 0.1 * s / self.T

    def change_direction(self, scatter_phi, scatter_theta):
        # Changes the direction of the electron by dphi and dtheta
        # Warning: wrong, transformation into local system is missing!
        
        # Current direction vector
        vx = sin(self.dtheta) * cos(self.dphi)
        vy = sin(self.dtheta) * sin(self.dphi)
        vz = cos(self.dtheta)
        v = np.array([vx, vy, vz])

        # Determine orthogonal basis (u,w)
        if abs(sin(self.dtheta)) < 1e-6:
            # Use standard axis if electron moves in z-direction
            u = np.array([1.0, 0.0, 0.0])
            w = np.array([0.0, 1.0, 0.0])
        else:
            u = np.array([-sin(self.dphi), cos(self.dphi), 0.0])
            # w is orthogonal to v and u
            w = np.cross(v, u)
            w /= np.linalg.norm(w)

        # New direction -> v is rotated by the angle "scatter_theta" (azimutal angle) in the surface (u,w)
        v_new = v * cos(scatter_theta) + (u * cos(scatter_phi) + w * sin(scatter_phi)) * sin(scatter_theta)
        # Norm
        v_new /= np.linalg.norm(v_new) 

        # Update angles
        self.dtheta = np.arccos(v_new[2])
        self.dphi = np.arctan2(v_new[1], v_new[0])

    def scatter(self, other):
        # Elastic scattering of electron with other electron, the angle by which the electron gets deflected is chosen randomly (this is correct!
        P = sqrt((self.T + self.m)**2 - self.m**2)
        phi_scatter = random.uniform(-pi, pi)
        theta_max = pi/2
        A = cos(theta_max/2) / sin(theta_max/2) 
        B = 1.0 
        u = random.uniform(0, 1)
        theta_cms = 2 * atan(1 / (u * (A - B) + B))

    
        theta1 = theta_cms / 2
        theta2 = theta_cms / 2

    
        self.change_direction(phi_scatter, theta1)
        other.change_direction(-phi_scatter, theta2)

        T1 = self.T * (cos(theta_cms/2)**2)
        T2 = self.T - T1

        self.T = T1
        other.T = T2