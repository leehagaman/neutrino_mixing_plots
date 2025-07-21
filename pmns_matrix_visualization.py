import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
import os
from nufit_params import get_PMNS, theta12, theta23, theta13, deltaCP

def create_unit_circle_plot(U):
    """
    Create unit circle visualization with arrows touching the circle edge
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    # Matrix element labels
    labels = [['U_{e1}', 'U_{e2}', 'U_{e3}'],
             ['U_{\mu 1}', 'U_{\mu 2}', 'U_{\mu 3}'],
             ['U_{\\tau 1}', 'U_{\\tau 2}', 'U_{\\tau 3}']]
    
    # Matrix element labels
    row_labels = ['e', 'μ', 'τ']
    col_labels = ['1', '2', '3']
    
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            
            # Get complex number
            z = U[i, j]
            magnitude = abs(z)
            phase = np.angle(z)
            
            # Draw unit circle
            circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
            ax.add_patch(circle)
            
            # Draw arrow to actual complex number position
            arrow_color = ["tab:purple", "tab:cyan", "tab:pink"][j]

            adjusted_magnitude = magnitude - 0.1
            
            ax.arrow(0, 0, adjusted_magnitude * np.cos(phase), adjusted_magnitude * np.sin(phase),
                    head_width=0.08, head_length=0.08, 
                    fc=arrow_color, ec=arrow_color, linewidth=3)
            
            """# Add magnitude circle to show actual magnitude
            if magnitude < 0.99:  # Small tolerance for floating point
                mag_circle = Circle((0, 0), magnitude, fill=False, color='blue', 
                                  linewidth=2, linestyle='--', alpha=0.8)
                ax.add_patch(mag_circle)"""
            
            # Set equal aspect ratio and limits
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove black boxes around subplots
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add row labels on the left edge
            if j == 0:
                ax.text(-1.8, 0, row_labels[i], ha='center', va='center', fontsize=22)
            
            # Add column labels on the top edge
            if i == 0:
                ax.text(0, 1.8, col_labels[j], ha='center', va='center', fontsize=22)
    
    plt.tight_layout()
    plt.savefig('plots/pmns_unit_circles.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved: plots/pmns_unit_circles.pdf")

def create_complex_values_plot(U):
    """
    Create a numerical display of complex values
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Matrix element labels
    row_labels = ['e', 'μ', 'τ']
    col_labels = ['1', '2', '3']
    
    # Create text display of matrix
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Draw matrix brackets
    bracket_min_x = 0.05
    bracket_max_x = 0.95
    bracket_min_y = 0.15
    bracket_max_y = 0.9
    bracket_width = 0.05
    lw=5
    # Left bracket
    ax.plot([bracket_min_x, bracket_min_x], [bracket_min_y, bracket_max_y], 'k-', linewidth=lw)
    ax.plot([bracket_min_x, bracket_min_x + bracket_width], [bracket_min_y, bracket_min_y], 'k-', linewidth=lw)
    ax.plot([bracket_min_x, bracket_min_x + bracket_width], [bracket_max_y, bracket_max_y], 'k-', linewidth=lw)
    # Right bracket
    ax.plot([bracket_max_x, bracket_max_x], [bracket_min_y, bracket_max_y], 'k-', linewidth=lw)
    ax.plot([bracket_max_x - bracket_width, bracket_max_x], [bracket_min_y, bracket_min_y], 'k-', linewidth=lw)
    ax.plot([bracket_max_x - bracket_width, bracket_max_x], [bracket_max_y, bracket_max_y], 'k-', linewidth=lw)
    
    # Add matrix elements
    for i in range(3):
        for j in range(3):
            z = U[i, j]
            
            # Position for this element
            x_offset = 0.2
            x_step = 0.28
            y_offset = 0.8
            y_step = 0.25
            x = x_offset + j * x_step
            y = y_offset - i * y_step
            
            # Format complex number
            if abs(z.imag) < 1e-10:  # Essentially real
                text = f'{z.real:.4f}'
            else:
                if z.real == 0:
                    if z.imag > 0:
                        text = f'{z.imag:.3f}i'
                    else:
                        text = f'{z.imag:.3f}i'
                else:
                    if z.imag >= 0:
                        text = f'{z.real:.3f} + {z.imag:.3f}i'
                    else:
                        text = f'{z.real:.3f} - {abs(z.imag):.3f}i'
            
            ax.text(x, y, text, ha='center', va='center', fontsize=16, family='monospace', fontweight='bold')
    
    # Add row and column labels
    for i, label in enumerate(row_labels):
        ax.text(0, y_offset - i * y_step, label, ha='center', va='center', fontsize=18, fontweight='bold')
    
    for j, label in enumerate(col_labels):
        ax.text(x_offset + j * x_step, 1, label, ha='center', va='center', fontsize=18, fontweight='bold')
    
    plt.savefig('plots/pmns_complex_values.pdf', bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.close()
    print("Saved: plots/pmns_complex_values.pdf")

def create_magnitude_plot(U):
    """
    Create a visualization showing only the magnitudes
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract magnitudes
    magnitudes = np.abs(U)
    
    # Create heatmap
    im = ax.imshow(magnitudes, cmap='viridis', aspect='equal', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{magnitudes[i,j]:.3f}', ha='center', va='center',
                    color='white' if magnitudes[i,j] < 0.5 else 'black', fontweight='bold', fontsize=18)
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['e', 'μ', 'τ'])
    ax.set_xlabel('Mass eigenstate')
    ax.set_ylabel('Flavor eigenstate')
    
    plt.tight_layout()
    plt.savefig('plots/pmns_magnitudes.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved: plots/pmns_magnitudes.pdf")

def create_pmns_visualization():
    """
    Create comprehensive PMNS matrix visualizations and save to PDF files
    """
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Generate PMNS matrix using NuFit parameters
    # Set Majorana phases to 0 for simplicity
    alpha1, alpha2 = 0, 0
    U = get_PMNS(theta12, theta23, theta13, deltaCP, alpha1, alpha2)
    
    # Create all three visualizations
    create_unit_circle_plot(U)
    create_complex_values_plot(U)
    create_magnitude_plot(U)
    
    return U

def print_matrix_info(U):
    """Print additional information about the PMNS matrix"""
    print("\nPMNS Matrix (3x3):")
    print("==================")
    print(f"Matrix shape: {U.shape}")
    print(f"Matrix dtype: {U.dtype}")
    print("\nFull matrix:")
    for i, row in enumerate(U):
        print(f"Row {i+1}: [{', '.join([f'{z:.6f}' for z in row])}]")
    
    print(f"\nUnitarity check: U†U = I?")
    product = U @ U.conj().T
    print("U†U =")
    for row in product:
        print(f"[{', '.join([f'{z:.2e}' for z in row])}]")
    
    print(f"\nColumn magnitude sums (should all be 1):")
    for j in range(3):
        col_sum = sum(abs(U[i, j])**2 for i in range(3))
        print(f"Column {j+1}: {col_sum:.6f}")

if __name__ == "__main__":
    print("Creating PMNS matrix visualization...")
    U = create_pmns_visualization()
    print("\nAll plots saved as PDF files:")
    print("- plots/pmns_unit_circles.pdf: Unit circle representation")
    print("- plots/pmns_complex_values.pdf: Numerical complex values")
    print("- plots/pmns_magnitudes.pdf: Element magnitudes")
    print_matrix_info(U) 