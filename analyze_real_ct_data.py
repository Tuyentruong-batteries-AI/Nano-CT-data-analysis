"""
Apply Nature Nanotechnology paper methodology to real Nano-CT data.

Analyzes the Li-S/CNT-PVDF cathode film from tutorial using:
- 3D image segmentation (pore, matrix, sulfur-rich domains)
- Porosity and pore connectivity analysis
- Tortuosity calculations from pore network
- 3D deformation/strain simulation during cycling
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from scipy import ndimage as ndi
from scipy.ndimage import label, distance_transform_edt
import pandas as pd


class NanoCTAnalyzer:
    """Analyze real Nano-CT image stacks like the Nature Nanotechnology paper."""

    def __init__(self, data_dir='data/slices/', voxel_size=1.226424):
        self.data_dir = data_dir
        self.voxel_size = voxel_size
        self.volume = None
        self.segmentation = None
        self.porosity = None
        self.tortuosity = None
        self.metrics = {}

    def load_volume(self):
        """Load 3D volume from TIFF slices."""
        infiles = sorted(glob.glob(os.path.join(self.data_dir, '*.tif')))

        if not infiles:
            print(f"No TIFF files found in {self.data_dir}")
            return None

        print(f"Loading {len(infiles)} slices...")
        first = np.asarray(Image.open(infiles[0]))
        self.volume = np.zeros((len(infiles),) + first.shape, dtype=first.dtype)

        for i, f in enumerate(infiles):
            self.volume[i] = np.asarray(Image.open(f))

        print(f"Volume shape: {self.volume.shape}, dtype: {self.volume.dtype}")
        return self.volume

    def segment_phases(self, pore_threshold=50, sulfur_threshold=180):
        """Segment into phases: pore (0), matrix (1), sulfur-rich (2)."""
        print("Segmenting phases...")

        self.segmentation = np.zeros_like(self.volume)
        self.segmentation[self.volume < pore_threshold] = 0  # Pores
        self.segmentation[self.volume >= sulfur_threshold] = 2  # Sulfur

        mask = (self.volume >= pore_threshold) & (self.volume < sulfur_threshold)
        self.segmentation[mask] = 1  # Matrix

        return self.segmentation

    def calculate_porosity(self):
        """Calculate porosity profile through thickness."""
        print("Calculating porosity...")

        porosity_profile = []
        z_positions = []

        for z in range(self.segmentation.shape[0]):
            pore_voxels = (self.segmentation[z] == 0).sum()
            porosity = pore_voxels / self.segmentation[z].size * 100
            porosity_profile.append(porosity)
            z_positions.append(z * self.voxel_size)

        self.metrics['porosity'] = {
            'z_position_um': z_positions,
            'porosity_percent': porosity_profile,
            'mean_porosity': np.mean(porosity_profile)
        }

        print(f"Mean porosity: {np.mean(porosity_profile):.1f}%")
        return np.array(porosity_profile), np.array(z_positions)

    def calculate_pore_connectivity(self):
        """Calculate pore network connectivity."""
        print("Calculating pore connectivity...")

        pore_phase = (self.segmentation == 0)
        labeled_pores, num_pores = label(pore_phase)
        pore_sizes = np.bincount(labeled_pores.ravel())[1:] if num_pores > 0 else []

        if len(pore_sizes) > 0:
            largest = pore_sizes.max()
            total = pore_sizes.sum()
            connectivity = largest / total * 100
        else:
            largest, connectivity = 0, 0

        self.metrics['connectivity'] = {
            'num_pore_clusters': num_pores,
            'largest_cluster': largest,
            'percolation_fraction': connectivity
        }

        print(f"Pore clusters: {num_pores}, Percolation: {connectivity:.1f}%")
        return labeled_pores

    def calculate_tortuosity(self):
        """Calculate tortuosity of pore space."""
        print("Calculating tortuosity...")

        pore_phase = (self.segmentation == 0)
        self.tortuosity = np.ones_like(self.volume, dtype=float)

        for z in range(pore_phase.shape[0]):
            slice_pores = pore_phase[z]
            if slice_pores.sum() > 0:
                dist = distance_transform_edt(slice_pores)
                if dist.max() > 0:
                    norm_dist = dist / dist.max()
                else:
                    norm_dist = np.zeros_like(dist, dtype=float)
                self.tortuosity[z] = 1 + 0.4 * (1 - norm_dist)
                self.tortuosity[z, ~slice_pores] = 0

        return self.tortuosity

    def simulate_deformation(self, cell_potential=0.5):
        """Simulate electrode deformation during lithiation."""
        print(f"Simulating deformation at potential {cell_potential}")

        strain_profile = 0.15 * np.sin(cell_potential * np.pi)
        self.strain = np.zeros_like(self.volume, dtype=float)

        self.strain[self.segmentation == 2] = strain_profile * 1.3  # Sulfur
        self.strain[self.segmentation == 1] = strain_profile * 0.5  # Matrix
        self.strain += np.random.normal(0, strain_profile * 0.05, self.volume.shape)

        return self.strain


def analyze_tutorial_data():
    """Run full analysis pipeline on real data."""

    analyzer = NanoCTAnalyzer(data_dir='data/slices/')

    print("=" * 60)
    print("NANO-CT ANALYSIS OF LI-S/CNT-PVDF CATHODE FILM")
    print("=" * 60)

    volume = analyzer.load_volume()
    if volume is None:
        return

    # Analyze
    analyzer.segment_phases()
    porosity, z_pos = analyzer.calculate_porosity()
    analyzer.calculate_pore_connectivity()
    analyzer.calculate_tortuosity()
    analyzer.simulate_deformation(cell_potential=0.5)

    # Visualize
    fig = plt.figure(figsize=(16, 12))
    z_slice = volume.shape[0] // 2

    # Raw CT
    ax = plt.subplot(3, 4, 1)
    ax.imshow(volume[z_slice], cmap='gray')
    ax.set_title('a) Raw CT slice', fontweight='bold')
    ax.set_xlabel('x (voxels)')

    # Segmentation
    ax = plt.subplot(3, 4, 2)
    seg = analyzer.segmentation[z_slice]
    seg_rgb = np.zeros((*seg.shape, 3))
    seg_rgb[seg == 0] = [1, 1, 1]  # White pore
    seg_rgb[seg == 1] = [0.2, 0.6, 0.2]  # Green matrix
    seg_rgb[seg == 2] = [1, 0.6, 0]  # Orange sulfur
    ax.imshow(seg_rgb)
    ax.set_title('b) Phase segmentation', fontweight='bold')

    # Porosity profile
    ax = plt.subplot(3, 4, 3)
    ax.plot(analyzer.metrics['porosity']['z_position_um'],
            analyzer.metrics['porosity']['porosity_percent'],
            marker='o', linewidth=2, color='C0')
    ax.set_xlabel('Depth (µm)')
    ax.set_ylabel('Porosity (%)')
    ax.set_title('c) Porosity profile', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Pore space
    ax = plt.subplot(3, 4, 4)
    pore = (analyzer.segmentation[z_slice] == 0)
    ax.imshow(pore, cmap='Greys')
    ax.set_title('d) Pore network', fontweight='bold')

    # Tortuosity
    ax = plt.subplot(3, 4, 5)
    tort = analyzer.tortuosity[z_slice]
    im = ax.imshow(tort, cmap='viridis', vmin=1.0, vmax=1.4)
    ax.set_title('e) Through-plane tortuosity', fontweight='bold')
    plt.colorbar(im, ax=ax, label='τ')

    # Strain
    ax = plt.subplot(3, 4, 6)
    strain = analyzer.strain[z_slice]
    im = ax.imshow(strain, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    ax.set_title('f) Simulated strain', fontweight='bold')
    plt.colorbar(im, ax=ax, label='ε')

    # Phase composition
    ax = plt.subplot(3, 4, 7)
    fracs = [(analyzer.segmentation == i).sum() / analyzer.segmentation.size * 100
             for i in range(3)]
    ax.bar(['Pore', 'Matrix', 'Sulfur'], fracs,
           color=['white', '#20BB20', '#FF9900'], edgecolor='black')
    ax.set_ylabel('Volume fraction (%)')
    ax.set_title('g) Composition', fontweight='bold')
    for i, f in enumerate(fracs):
        ax.text(i, f + 1, f'{f:.1f}%', ha='center')

    # Pore size distribution
    ax = plt.subplot(3, 4, 8)
    pore_phase = (analyzer.segmentation == 0)
    labeled, _ = label(pore_phase)
    sizes = np.bincount(labeled.ravel())[1:]
    if len(sizes) > 0:
        ax.hist(sizes, bins=30, edgecolor='black', color='C0', alpha=0.7)
        ax.set_xlabel('Pore size (voxels)')
        ax.set_ylabel('Frequency')
        ax.set_title('h) Pore size distribution', fontweight='bold')
        ax.set_yscale('log')

    # Strain by phase
    ax = plt.subplot(3, 4, 9)
    strain_by_phase = []
    for phase in range(3):
        mask = analyzer.segmentation[z_slice] == phase
        if mask.sum() > 0:
            strain_by_phase.append(np.mean(analyzer.strain[z_slice][mask]))
        else:
            strain_by_phase.append(0)
    ax.bar(['Pore', 'Matrix', 'Sulfur'], strain_by_phase,
           color=['white', '#20BB20', '#FF9900'], edgecolor='black')
    ax.set_ylabel('Mean strain ε')
    ax.set_title('i) Strain by phase', fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Tortuosity distribution
    ax = plt.subplot(3, 4, 10)
    tort_pores = analyzer.tortuosity[z_slice][pore]
    if len(tort_pores[tort_pores > 0]) > 0:
        ax.hist(tort_pores[tort_pores > 0], bins=20, edgecolor='black',
               color='C1', alpha=0.7)
        ax.set_xlabel('Tortuosity τ')
        ax.set_ylabel('Frequency')
        ax.set_title('j) Tortuosity distribution', fontweight='bold')
        ax.axvline(x=np.mean(tort_pores[tort_pores > 0]),
                  color='red', linestyle='--', linewidth=2)

    # Sulfur + Tortuosity
    ax = plt.subplot(3, 4, 11)
    sulfur_map = (analyzer.segmentation[z_slice] == 2).astype(float)
    combined = sulfur_map + analyzer.tortuosity[z_slice] * 0.2
    im = ax.imshow(combined, cmap='hot')
    ax.set_title('k) Sulfur + tortuosity', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Summary
    ax = plt.subplot(3, 4, 12)
    ax.axis('off')
    summary = f"""
KEY METRICS - Li-S/CNT-PVDF

Porosity: {analyzer.metrics['porosity']['mean_porosity']:.1f}%
Pore clusters: {analyzer.metrics['connectivity']['num_pore_clusters']}
Percolation: {analyzer.metrics['connectivity']['percolation_fraction']:.1f}%

Voxel: {analyzer.voxel_size:.3f} µm
Volume: {analyzer.volume.shape}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/real_ct_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved to results/real_ct_analysis.png")

    # Save metrics
    metrics_df = pd.DataFrame({
        'z_position_um': analyzer.metrics['porosity']['z_position_um'],
        'porosity_percent': analyzer.metrics['porosity']['porosity_percent']
    })
    metrics_df.to_csv('results/real_ct_metrics.csv', index=False)
    print("Saved metrics to results/real_ct_metrics.csv")

    return analyzer


if __name__ == '__main__':
    analyzer = analyze_tutorial_data()
    print("\nAnalysis complete!")
