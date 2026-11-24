"""
Battery electrode mechanics and ionic diffusion analysis.

Generates synthetic data similar to Nature Nanotechnology paper:
"Four-dimensional X-ray nano-holotomography reveals multiscale
chemomechanics in composite graphite-silicon battery electrodes"

Covers:
- Kinematic fields (displacement, strain) during battery cycling
- Ionic diffusion pathways and tortuosity calculations
- 3D visualization of electrode deformation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.sparse.linalg import cg
import pandas as pd


class BatteryElectrode:
    """Synthetic battery electrode with deformation and diffusion properties."""

    def __init__(self, shape=(100, 100, 100), voxel_size=1.0):
        """
        Initialize electrode volume.

        Args:
            shape: 3D array shape (z, y, x)
            voxel_size: physical size per voxel (µm)
        """
        self.shape = shape
        self.voxel_size = voxel_size
        self.volume = np.zeros(shape)
        self.displacement = np.zeros((3,) + shape)  # (x, y, z displacements)
        self.strain = np.zeros(shape)  # volumetric strain
        self.tortuosity = np.zeros(shape)

    def create_composite_structure(self, graphite_fraction=0.6):
        """Create synthetic graphite-silicon composite structure."""
        z, y, x = self.shape

        # Graphite particles (larger, clustered)
        for _ in range(50):
            cy, cx = np.random.randint(10, y-10), np.random.randint(10, x-10)
            radius = np.random.randint(5, 15)
            yy, xx = np.ogrid[:y, :x]
            mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
            self.volume[:, mask] = 1  # Graphite phase

        # Silicon particles (smaller, distributed)
        for _ in range(100):
            cz, cy, cx = (np.random.randint(10, z-10),
                         np.random.randint(10, y-10),
                         np.random.randint(10, x-10))
            radius = np.random.randint(2, 6)
            zz, yy, xx = np.ogrid[:z, :y, :x]
            mask = ((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2) <= radius**2
            self.volume[mask] = 2  # Silicon phase

        return self.volume

    def add_deformation_during_lithiation(self, cell_potential=0.1, max_strain=0.15):
        """
        Simulate deformation during lithiation.

        Args:
            cell_potential: normalized cell potential (0-1)
            max_strain: maximum volumetric strain
        """
        z, y, x = self.shape

        # Strain increases then decreases (lithiation -> delithiation)
        # Based on Fig. 3c, 3d pattern
        strain_profile = max_strain * np.sin(cell_potential * np.pi)

        # Silicon expands more than graphite
        for idx in np.argwhere(self.volume == 2):  # Silicon
            self.strain[tuple(idx)] = strain_profile * 1.2
        for idx in np.argwhere(self.volume == 1):  # Graphite
            self.strain[tuple(idx)] = strain_profile * 0.3

        # Add spatial variations (non-uniform deformation)
        noise = np.random.normal(0, strain_profile * 0.1, self.shape)
        self.strain += noise

        return self.strain

    def calculate_displacement_field(self):
        """Calculate 3D displacement from strain field."""
        # Displacement proportional to position and strain gradient
        z, y, x = np.indices(self.shape)

        # x-displacement
        self.displacement[0] = self.strain * (x - x.mean()) / 50
        # y-displacement
        self.displacement[1] = self.strain * (y - y.mean()) / 50
        # z-displacement
        self.displacement[2] = self.strain * (z - z.mean()) / 50

        # Add smoothing
        from scipy import ndimage
        for i in range(3):
            self.displacement[i] = ndimage.gaussian_filter(
                self.displacement[i], sigma=2
            )

        return self.displacement

    def calculate_tortuosity(self):
        """
        Calculate tortuosity of pores (space not occupied by particles).

        Through-plane and in-plane tortuosity metrics.
        """
        # Pore space (inverse of particles)
        pore_space = (self.volume == 0).astype(float)

        # Through-plane tortuosity (z-direction)
        for z in range(self.shape[0]):
            slice_pores = pore_space[z]
            if slice_pores.sum() > 0:
                # Increased tortuosity near particles
                particle_proximity = ndi.distance_transform_edt(
                    slice_pores < 0.5
                )
                self.tortuosity[z] = 1 + 0.3 / (1 + particle_proximity)

        return self.tortuosity

    def calculate_ionic_flux_paths(self):
        """
        Calculate high-probability ionic diffusion pathways.
        Highest flux through regions of lowest tortuosity.
        """
        flux = np.zeros(self.shape)

        # High flux through electrodes (Si regions)
        flux[self.volume == 2] = 0.8
        flux[self.volume == 1] = 0.5

        # Smooth to show diffusion fronts
        from scipy import ndimage
        flux = ndimage.gaussian_filter(flux, sigma=3)

        return flux

    def get_metrics_vs_potential(self, potentials=np.linspace(0, 1, 11)):
        """Calculate electrode metrics across full lithiation cycle."""
        metrics = {
            'potential': [],
            'strain_mean': [],
            'strain_std': [],
            'displacement_magnitude': [],
            'tortuosity_mean': []
        }

        for pot in potentials:
            self.add_deformation_during_lithiation(pot)
            self.calculate_displacement_field()
            self.calculate_tortuosity()

            disp_mag = np.sqrt(
                self.displacement[0]**2 +
                self.displacement[1]**2 +
                self.displacement[2]**2
            )

            metrics['potential'].append(pot)
            metrics['strain_mean'].append(np.mean(self.strain))
            metrics['strain_std'].append(np.std(self.strain))
            metrics['displacement_magnitude'].append(np.mean(disp_mag))
            metrics['tortuosity_mean'].append(np.mean(self.tortuosity))

        return pd.DataFrame(metrics)


class BatteryMechanicsAnalysis:
    """Analysis and visualization of battery electrode mechanics."""

    def __init__(self, electrode):
        self.electrode = electrode

    def plot_strain_vs_potential(self, ax=None):
        """Plot strain metrics vs cell potential (like Fig. 3c)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        metrics = self.electrode.get_metrics_vs_potential()

        ax.errorbar(
            metrics['potential'],
            metrics['strain_mean'],
            yerr=metrics['strain_std'],
            marker='o', linewidth=2, markersize=6,
            label='Volumetric strain', color='C0'
        )
        ax.set_xlabel('Cell potential (V)', fontsize=11)
        ax.set_ylabel('Strain (ε, %)', fontsize=11)
        ax.set_title('Electrode volumetric strain during lithiation', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax

    def plot_displacement_field(self, z_slice=50, ax=None):
        """Plot 2D displacement field at a slice (like Fig. 3a)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        # Calculate magnitude of in-plane displacement
        disp_mag = np.sqrt(
            self.electrode.displacement[0, z_slice]**2 +
            self.electrode.displacement[1, z_slice]**2
        )

        im = ax.imshow(
            disp_mag, cmap='RdYlBu_r', origin='lower',
            extent=[0, self.electrode.shape[2], 0, self.electrode.shape[1]]
        )
        ax.set_xlabel('x (voxels)', fontsize=11)
        ax.set_ylabel('y (voxels)', fontsize=11)
        ax.set_title(f'Displacement magnitude at z={z_slice}', fontsize=12)
        plt.colorbar(im, ax=ax, label='Displacement (µm)')

        return ax

    def plot_tortuosity_map(self, z_slice=50, ax=None):
        """Plot tortuosity map (like Fig. 4c)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        tortuosity_slice = self.electrode.tortuosity[z_slice]

        im = ax.imshow(
            tortuosity_slice, cmap='viridis', origin='lower',
            extent=[0, self.electrode.shape[2], 0, self.electrode.shape[1]],
            vmin=1.0, vmax=1.5
        )
        ax.set_xlabel('x (voxels)', fontsize=11)
        ax.set_ylabel('y (voxels)', fontsize=11)
        ax.set_title(f'Ionic diffusion tortuosity at z={z_slice}', fontsize=12)
        cbar = plt.colorbar(im, ax=ax, label='Tortuosity (τ)')

        return ax

    def plot_tortuosity_vs_potential(self, ax=None):
        """Plot tortuosity evolution vs cell potential (like Fig. 4e)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        metrics = self.electrode.get_metrics_vs_potential()

        ax.plot(
            metrics['potential'],
            metrics['tortuosity_mean'],
            marker='s', linewidth=2, markersize=6,
            label='Through-plane tortuosity', color='C1'
        )
        ax.set_xlabel('Cell potential (V)', fontsize=11)
        ax.set_ylabel('Tortuosity (τ)', fontsize=11)
        ax.set_title('Ionic diffusion tortuosity during cell cycling', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([1.0, 2.0])

        return ax

    def plot_structure_and_phase_map(self, z_slice=50, ax=None):
        """Plot electrode phase composition (like Fig. 1c)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        phase_map = self.electrode.volume[z_slice]

        cmap = plt.cm.colors.ListedColormap(
            ['white', '#1f77b4', '#ff7f0e']  # Pore, graphite, silicon
        )

        im = ax.imshow(
            phase_map, cmap=cmap, origin='lower',
            extent=[0, self.electrode.shape[2], 0, self.electrode.shape[1]]
        )
        ax.set_xlabel('x (voxels)', fontsize=11)
        ax.set_ylabel('y (voxels)', fontsize=11)
        ax.set_title(f'Phase composition at z={z_slice}', fontsize=12)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Pore'),
            Patch(facecolor='#1f77b4', label='Graphite'),
            Patch(facecolor='#ff7f0e', label='Silicon')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        return ax


def generate_figure_3_equivalent(save_path=None):
    """Generate Figure 3 equivalent: kinematic fields during lithiation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Fig. 3 Equivalent: Kinematic fields during Gr-Si lithiation',
        fontsize=14, fontweight='bold'
    )

    electrode = BatteryElectrode(shape=(60, 100, 100))
    electrode.create_composite_structure()
    analysis = BatteryMechanicsAnalysis(electrode)

    # Subplot a: Structure and phase
    analysis.plot_structure_and_phase_map(z_slice=30, ax=axes[0, 0])

    # Subplot b: Displacement field
    electrode.add_deformation_during_lithiation(cell_potential=0.5)
    electrode.calculate_displacement_field()
    analysis.plot_displacement_field(z_slice=30, ax=axes[0, 1])

    # Subplot c: Strain vs potential
    analysis.plot_strain_vs_potential(ax=axes[1, 0])

    # Subplot d: Displacement magnitude vs potential
    potentials = np.linspace(0, 1, 11)
    metrics = electrode.get_metrics_vs_potential(potentials)
    axes[1, 1].plot(
        metrics['potential'],
        metrics['displacement_magnitude'],
        marker='o', linewidth=2, markersize=6, color='C2'
    )
    axes[1, 1].set_xlabel('Cell potential (V)', fontsize=11)
    axes[1, 1].set_ylabel('Mean displacement (µm)', fontsize=11)
    axes[1, 1].set_title('Electrode displacement magnitude', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def generate_figure_4_equivalent(save_path=None):
    """Generate Figure 4 equivalent: tortuosity and ionic diffusion."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Fig. 4 Equivalent: Ionic diffusion pathways and tortuosity',
        fontsize=14, fontweight='bold'
    )

    electrode = BatteryElectrode(shape=(60, 100, 100))
    electrode.create_composite_structure()
    analysis = BatteryMechanicsAnalysis(electrode)

    # Initialize at mid-potential for visualization
    electrode.add_deformation_during_lithiation(cell_potential=0.5)
    electrode.calculate_tortuosity()

    # Subplot a: Structure
    analysis.plot_structure_and_phase_map(z_slice=30, ax=axes[0, 0])

    # Subplot b: Tortuosity map
    analysis.plot_tortuosity_map(z_slice=30, ax=axes[0, 1])

    # Subplot c: Tortuosity vs potential
    analysis.plot_tortuosity_vs_potential(ax=axes[1, 0])

    # Subplot d: Displacement vs tortuosity correlation
    potentials = np.linspace(0, 1, 11)
    metrics = electrode.get_metrics_vs_potential(potentials)
    axes[1, 1].scatter(
        metrics['displacement_magnitude'],
        metrics['tortuosity_mean'],
        s=100, alpha=0.6, c=metrics['potential'], cmap='cool'
    )
    axes[1, 1].set_xlabel('Mean displacement (µm)', fontsize=11)
    axes[1, 1].set_ylabel('Mean tortuosity (τ)', fontsize=11)
    axes[1, 1].set_title('Deformation-tortuosity correlation', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='cool',
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1, 1])
    cbar.set_label('Cell potential (V)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def generate_figure_6_equivalent(save_path=None):
    """Generate Figure 6 equivalent: correlative maps of parameters impacting electrochemical activity."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(
        'Fig. 6 Equivalent: Factors impacting electrochemical activity in Gr-Si',
        fontsize=14, fontweight='bold'
    )

    electrode = BatteryElectrode(shape=(60, 100, 100))
    electrode.create_composite_structure()
    electrode.add_deformation_during_lithiation(cell_potential=0.5)
    electrode.calculate_tortuosity()

    z_slice = 30

    # Create parameter maps
    structure = electrode.volume[z_slice]
    strain_map = electrode.strain[z_slice]
    tortuosity_map = electrode.tortuosity[z_slice]

    # Calculate derived parameters
    si_proximity = np.zeros((electrode.shape[1], electrode.shape[2]))
    for y in range(electrode.shape[1]):
        for x in range(electrode.shape[2]):
            if structure[y, x] != 0:
                si_proximity[y, x] = 1

    # Smooth proximity
    si_proximity = ndi.gaussian_filter(si_proximity, sigma=5)

    electrochemical_activity = (
        0.3 * (tortuosity_map / (tortuosity_map.max() + 1e-6)) +
        0.4 * (1 - np.abs(strain_map) / (np.abs(strain_map).max() + 1e-6)) +
        0.3 * (si_proximity / (si_proximity.max() + 1e-6))
    )

    # Plot 1: Morphology - structure
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(structure, cmap='gray', origin='lower')
    ax1.set_title('a) Electrode structure', fontsize=11, fontweight='bold')
    ax1.set_xlabel('x (voxels)', fontsize=9)
    ax1.set_ylabel('y (voxels)', fontsize=9)

    # Plot 2: Ionic diffusion - tortuosity
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(tortuosity_map, cmap='viridis', origin='lower', vmin=1.0, vmax=1.5)
    ax2.set_title('b) Through-plane tortuosity', fontsize=11, fontweight='bold')
    ax2.set_xlabel('x (voxels)', fontsize=9)
    ax2.set_ylabel('y (voxels)', fontsize=9)

    # Plot 3: Mechanics - strain
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(strain_map, cmap='RdBu_r', origin='lower')
    ax3.set_title('c) Volumetric strain', fontsize=11, fontweight='bold')
    ax3.set_xlabel('x (voxels)', fontsize=9)
    ax3.set_ylabel('y (voxels)', fontsize=9)

    # Plot 4: Distance to nearest CBD-Si
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(si_proximity, cmap='hot', origin='lower')
    ax4.set_title('d) Proximity to CBD-Si', fontsize=11, fontweight='bold')
    ax4.set_xlabel('x (voxels)', fontsize=9)
    ax4.set_ylabel('y (voxels)', fontsize=9)

    # Plot 5: Electrochemical activity
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(electrochemical_activity, cmap='plasma', origin='lower')
    ax5.set_title('e) Predicted activity', fontsize=11, fontweight='bold')
    ax5.set_xlabel('x (voxels)', fontsize=9)
    ax5.set_ylabel('y (voxels)', fontsize=9)

    # Plot 6: Combined heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow((electrochemical_activity + strain_map/(np.abs(strain_map).max()+1e-6))/2,
                     cmap='coolwarm', origin='lower')
    ax6.set_title('f) Activity-strain map', fontsize=11, fontweight='bold')
    ax6.set_xlabel('x (voxels)', fontsize=9)
    ax6.set_ylabel('y (voxels)', fontsize=9)

    # Plot 7: Activity vs tortuosity
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(tortuosity_map.flatten(), electrochemical_activity.flatten(),
               alpha=0.3, s=20, c='C0')
    ax7.set_xlabel('Tortuosity (τ)', fontsize=10)
    ax7.set_ylabel('Activity', fontsize=10)
    ax7.set_title('g) Tortuosity-activity', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Activity vs strain
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(strain_map.flatten(), electrochemical_activity.flatten(),
               alpha=0.3, s=20, c='C1')
    ax8.set_xlabel('Strain (ε, %)', fontsize=10)
    ax8.set_ylabel('Activity', fontsize=10)
    ax8.set_title('h) Strain-activity', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Plot 9: Activity vs Si proximity
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.scatter(si_proximity.flatten(), electrochemical_activity.flatten(),
               alpha=0.3, s=20, c='C2')
    ax9.set_xlabel('Si proximity', fontsize=10)
    ax9.set_ylabel('Activity', fontsize=10)
    ax9.set_title('i) Proximity-activity', fontsize=11, fontweight='bold')
    ax9.grid(True, alpha=0.3)

    # Add colorbars
    for im, ax in [(im2, ax2), (im3, ax3), (im4, ax4), (im5, ax5), (im6, ax6)]:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


if __name__ == '__main__':
    # Generate sample data and plots
    print("Generating synthetic battery mechanics data...")

    # Figure 3 equivalent
    fig3 = generate_figure_3_equivalent('results/figure_3_equivalent.png')

    # Figure 4 equivalent
    fig4 = generate_figure_4_equivalent('results/figure_4_equivalent.png')

    # Figure 6 equivalent
    fig6 = generate_figure_6_equivalent('results/figure_6_equivalent.png')

    # Generate sample data CSV
    electrode = BatteryElectrode(shape=(60, 100, 100))
    electrode.create_composite_structure()
    metrics = electrode.get_metrics_vs_potential()
    metrics.to_csv('results/electrode_metrics.csv', index=False)
    print("\nMetrics saved to results/electrode_metrics.csv")

    print("Done! Check results/ folder for outputs.")
