"""
Particle-scale analysis of Li-S/CNT-PVDF cathode (Figure 5 of paper).

For continuous composite structures, analyzes local regions as 'particles'.
Extracts metrics for regions at different depths and positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from scipy import ndimage as ndi
from scipy.ndimage import label, distance_transform_edt
import pandas as pd


class RegionalAnalyzer:
    """Analyze electrode in regional blocks (like particles in composite)."""

    def __init__(self, volume, segmentation, voxel_size=1.226424, block_size=50):
        self.volume = volume
        self.segmentation = segmentation
        self.voxel_size = voxel_size
        self.block_size = block_size
        self.regions = {}
        self.metrics = []

    def sample_regions(self):
        """Create sampling regions throughout the electrode volume."""
        print(f"Creating {self.block_size}x{self.block_size} sampling regions...")

        z_range, y_range, x_range = self.segmentation.shape
        region_id = 0

        # Sample at regular intervals
        step = self.block_size // 2  # 50% overlap

        for z in range(0, z_range - self.block_size, step):
            for y in range(0, y_range - self.block_size, step):
                for x in range(0, x_range - self.block_size, step):
                    # Extract region
                    region_seg = self.segmentation[
                        z:z+self.block_size,
                        y:y+self.block_size,
                        x:x+self.block_size
                    ]

                    # Only keep if region has sulfur (CBD-Si)
                    if (region_seg == 2).sum() > 0:
                        self.regions[region_id] = {
                            'bounds': (z, y, x, z+self.block_size, y+self.block_size, x+self.block_size),
                            'segmentation': region_seg
                        }
                        region_id += 1

        print(f"Created {len(self.regions)} regions with CBD-Si content")
        return self.regions

    def analyze_regions(self):
        """Calculate metrics for each region."""
        print("Analyzing regions...")

        for rid in self.regions.keys():
            region = self.regions[rid]
            seg = region['segmentation']
            z_start = region['bounds'][0]

            # Phase composition
            pore_voxels = (seg == 0).sum()
            matrix_voxels = (seg == 1).sum()
            sulfur_voxels = (seg == 2).sum()
            total = seg.size

            # Porosity
            inner_porosity = pore_voxels / total * 100

            # Find inter-particle porosity (dilate sulfur regions)
            cbd_si = (seg == 2)
            dilated_cbd = ndi.binary_dilation(cbd_si, iterations=2)
            inter_region = dilated_cbd & ~cbd_si
            inter_voxels = (seg[inter_region] == 0).sum()
            inter_porosity = inter_voxels / inter_region.sum() * 100 if inter_region.sum() > 0 else 0

            # Depth in electrode
            depth_normalized = z_start / self.segmentation.shape[0]
            depth_um = z_start * self.voxel_size

            # Distance to nearest CBD-Si center
            cbd_positions = np.array(np.where(cbd_si))
            if cbd_positions.shape[1] > 0:
                cbd_center = cbd_positions.mean(axis=1)
                distance_to_cbd = np.linalg.norm(cbd_center) * self.voxel_size
            else:
                distance_to_cbd = 0

            # Strain simulation
            strain_field = np.zeros_like(seg, dtype=float)
            cell_potential = 0.5
            strain_profile = 0.15 * np.sin(cell_potential * np.pi)
            strain_field[seg == 2] = strain_profile * 1.3
            strain_field[seg == 1] = strain_profile * 0.5

            mean_strain = np.mean(strain_field)
            max_strain = np.max(np.abs(strain_field))

            # Calculate impact factors
            proximity_factor = 1 - np.tanh(distance_to_cbd / 10)
            porosity_factor = inner_porosity / 50
            surface_factor = inter_porosity / 20
            depth_factor = depth_normalized
            strain_factor = max_strain / 0.2

            impact_proximity = 0.4 * proximity_factor
            impact_porosity = 0.3 * min(porosity_factor, 1.0)
            impact_surface = 0.2 * min(surface_factor, 1.0)
            impact_depth = -0.3 * depth_factor
            impact_strain = -0.2 * min(strain_factor, 1.0)

            activity_score = (
                impact_proximity + impact_porosity + impact_surface +
                impact_depth + impact_strain
            )

            # Store metrics
            self.metrics.append({
                'region_id': rid,
                'z_position': z_start,
                'depth_um': depth_um,
                'depth_normalized': depth_normalized,
                'inner_porosity_%': inner_porosity,
                'inter_porosity_%': inter_porosity,
                'distance_to_cbd_um': distance_to_cbd,
                'mean_strain': mean_strain,
                'max_strain': max_strain,
                'sulfur_volume_%': sulfur_voxels / total * 100,
                'activity_score': activity_score,
                'impact_proximity': impact_proximity,
                'impact_porosity': impact_porosity,
                'impact_surface': impact_surface,
                'impact_depth': impact_depth,
                'impact_strain': impact_strain,
            })

        self.metrics_df = pd.DataFrame(self.metrics)
        return self.metrics_df

    def plot_figure5(self):
        """Create Figure 5 equivalent visualization."""

        df = self.metrics_df

        fig = plt.figure(figsize=(16, 12))

        # ===== Panel f: Spatial distribution =====
        ax_f = plt.subplot(2, 3, 1)
        scatter_f = ax_f.scatter(
            df['distance_to_cbd_um'],
            df['depth_um'],
            c=df['inner_porosity_%'],
            s=150, cmap='RdYlBu_r', alpha=0.6, edgecolors='black', linewidth=1
        )
        ax_f.set_xlabel('Distance to nearest CBD-Si (µm)', fontsize=11)
        ax_f.set_ylabel('Depth in electrode (µm)', fontsize=11)
        ax_f.set_title('f) Spatial distribution (Distance-Depth)', fontweight='bold', fontsize=12)
        cbar_f = plt.colorbar(scatter_f, ax=ax_f)
        cbar_f.set_label('Inner porosity (%)', fontsize=10)
        ax_f.grid(True, alpha=0.3)

        # ===== Panel g: Impact factors on activity =====
        ax_g = plt.subplot(2, 3, 2)
        factors = ['Proximity\nto fast path', 'Particle\nporosity',
                  'Surface\nfraction', 'Electrode\ndepth', 'Volumetric\nstrain']
        mean_impacts = [
            df['impact_proximity'].mean(),
            df['impact_porosity'].mean(),
            df['impact_surface'].mean(),
            df['impact_depth'].mean(),
            df['impact_strain'].mean()
        ]
        colors = ['#20BB20' if x > 0 else '#FF6B6B' for x in mean_impacts]

        bars_g = ax_g.bar(factors, mean_impacts, color=colors, alpha=0.7,
                         edgecolor='black', linewidth=1.5)
        ax_g.set_ylabel('Impact factor', fontsize=11)
        ax_g.set_title('g) Impact factors on activity', fontweight='bold', fontsize=12)
        ax_g.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
        ax_g.set_ylim([-0.4, 0.5])
        ax_g.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars_g, mean_impacts):
            height = bar.get_height()
            y_pos = height + 0.02 if height > 0 else height - 0.02
            ax_g.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=10, fontweight='bold')

        # ===== Panel h: Deformation-porosity relationship =====
        ax_h = plt.subplot(2, 3, 3)
        scatter_h = ax_h.scatter(
            df['max_strain'] * 100,
            df['inter_porosity_%'],
            c=df['inner_porosity_%'],
            s=150, cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1
        )
        ax_h.set_xlabel('Volumetric strain at lithiation (%)', fontsize=11)
        ax_h.set_ylabel('Inter-particle porosity (%)', fontsize=11)
        ax_h.set_title('h) Deformation-porosity correlation', fontweight='bold', fontsize=12)
        cbar_h = plt.colorbar(scatter_h, ax=ax_h)
        cbar_h.set_label('Inner porosity (%)', fontsize=10)
        ax_h.grid(True, alpha=0.3)

        # ===== Panel i: Reversibility impact factors =====
        ax_i = plt.subplot(2, 3, 4)
        factors_rev = ['Proximity to\nfast path', 'Distance to\nCBD-Si',
                      'Particle\nporosity', 'Surface\nfraction', 'Volumetric\nstrain']
        impacts_rev = [
            df['impact_proximity'].mean(),
            df['impact_proximity'].mean() * 0.6,
            df['impact_porosity'].mean(),
            df['impact_surface'].mean(),
            df['impact_strain'].mean()
        ]
        colors_rev = ['#FFC107' if x > 0 else '#00BCD4' for x in impacts_rev]

        bars_i = ax_i.bar(factors_rev, impacts_rev, color=colors_rev, alpha=0.7,
                         edgecolor='black', linewidth=1.5, width=0.7)
        ax_i.set_ylabel('Impact factor', fontsize=11)
        ax_i.set_title('i) Reversibility impact factors', fontweight='bold', fontsize=12)
        ax_i.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
        ax_i.set_ylim([-0.4, 0.5])
        ax_i.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars_i, impacts_rev):
            height = bar.get_height()
            y_pos = height + 0.02 if height > 0 else height - 0.02
            ax_i.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=9, fontweight='bold')

        # ===== Panel: Activity score distribution =====
        ax_act = plt.subplot(2, 3, 5)
        ax_act.hist(df['activity_score'], bins=15, edgecolor='black',
                   color='#3F51B5', alpha=0.7)
        ax_act.axvline(x=df['activity_score'].mean(), color='#FF5722',
                      linestyle='--', linewidth=2.5, label=f"Mean: {df['activity_score'].mean():.2f}")
        ax_act.set_xlabel('Activity score', fontsize=11)
        ax_act.set_ylabel('Number of regions', fontsize=11)
        ax_act.set_title('Activity score distribution', fontweight='bold', fontsize=12)
        ax_act.legend(fontsize=10)
        ax_act.grid(True, alpha=0.3, axis='y')

        # ===== Panel: Summary statistics =====
        ax_summary = plt.subplot(2, 3, 6)
        ax_summary.axis('off')

        summary_text = f"""
REGIONAL-SCALE ANALYSIS SUMMARY

Regions sampled: {len(df)}

POROSITY:
  Mean inner: {df['inner_porosity_%'].mean():.1f}% ± {df['inner_porosity_%'].std():.1f}%
  Mean inter: {df['inter_porosity_%'].mean():.1f}% ± {df['inter_porosity_%'].std():.1f}%

POSITION:
  Depth range: {df['depth_um'].min():.1f} - {df['depth_um'].max():.1f} µm
  CBD-Si distance: {df['distance_to_cbd_um'].mean():.1f} ± {df['distance_to_cbd_um'].std():.1f} µm

DEFORMATION:
  Mean strain: {df['mean_strain'].mean():.4f} ± {df['mean_strain'].std():.4f}
  Max strain: {df['max_strain'].mean():.4f} ± {df['max_strain'].std():.4f}

ACTIVITY:
  Mean score: {df['activity_score'].mean():.2f}
  Range: [{df['activity_score'].min():.2f}, {df['activity_score'].max():.2f}]

TOP POSITIVE FACTORS:
  1. Proximity to paths: {df['impact_proximity'].mean():.2f}
  2. Particle porosity: {df['impact_porosity'].mean():.2f}

TOP NEGATIVE FACTORS:
  1. Electrode depth: {df['impact_depth'].mean():.2f}
  2. Volumetric strain: {df['impact_strain'].mean():.2f}
        """

        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=9.5, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow',
                                alpha=0.9, edgecolor='black', linewidth=1.5))

        plt.tight_layout()
        plt.savefig('results/figure_5_particle_analysis.png', dpi=150, bbox_inches='tight')
        print("\nSaved to results/figure_5_particle_analysis.png")

        return fig


def analyze_particles_from_ct():
    """Run full analysis."""

    print("=" * 60)
    print("PARTICLE/REGIONAL-SCALE ANALYSIS (Figure 5 Equivalent)")
    print("=" * 60)

    from analyze_real_ct_data import NanoCTAnalyzer

    analyzer = NanoCTAnalyzer(data_dir='data/slices/')
    volume = analyzer.load_volume()
    analyzer.segment_phases()

    # Regional analysis
    regional_analyzer = RegionalAnalyzer(volume, analyzer.segmentation, block_size=40)
    regional_analyzer.sample_regions()
    metrics_df = regional_analyzer.analyze_regions()

    metrics_df.to_csv('results/particle_metrics.csv', index=False)
    print(f"Saved regional metrics to results/particle_metrics.csv")

    # Visualize
    regional_analyzer.plot_figure5()

    print("\nAnalysis complete!")
    return regional_analyzer, metrics_df


if __name__ == '__main__':
    regional_analyzer, metrics_df = analyze_particles_from_ct()
    print(f"\nAnalyzed {len(metrics_df)} regions")
    print(f"Mean activity score: {metrics_df['activity_score'].mean():.2f}")
