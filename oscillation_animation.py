import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import nufit_params

def setup_experiment_plot(ax, delta_m21_sq, delta_m31_sq):
    """Add oscillation lines and experiment boxes to the L vs E plot."""
    
    # Add oscillation lines from the experiment plot
    x_values = np.array([1, 1e8])
    ax.plot(x_values, delta_m21_sq * x_values, 'k--', linewidth=2, alpha=0.7)
    ax.text(1e5, 15, r'$\Delta m_{21}^2$-driven first maximum', rotation=37, fontsize=10, alpha=0.7)
    ax.plot(x_values, delta_m31_sq * x_values, 'k--', linewidth=2, alpha=0.7)
    ax.text(3e3, 15, r'$\Delta m_{31}^2$-driven first maximum', rotation=37, fontsize=10, alpha=0.7)
    
    # Helper function to add experiment boxes
    def add_box(ax, xmin, xmax, ymin, ymax, color, text, text_pos=None, rotation=0, edgecolor=None):
        edgecolor = edgecolor or color
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=color, edgecolor=edgecolor, lw=2, alpha=0.7))
        tx, ty = text_pos if text_pos else ((xmin + xmax) / 2, (ymin + ymax) / 2)
        ax.text(tx, ty, text, fontsize=8, color=edgecolor, rotation=rotation, ha='center', va='center')
    
    # Atmospheric neutrinos
    add_box(ax, 3e6, 1.2e7, 1e4, 1e6, 'xkcd:peach', r'Atmospheric $\nu$', text_pos=(6.3e6, 1.2e5), rotation=90, edgecolor='xkcd:orange')
    
    # Long baseline experiments
    add_box(ax, 1.3e6, 1.3e6*1.05, 100, 5e3, 'xkcd:red', r'DUNE', text_pos=(1.3e6, 1.3e4), edgecolor='xkcd:red', rotation=90)
    add_box(ax, 810e3, 810e3*1.04, 0.5e3, 5e3, 'xkcd:light orange', r'NO$\nu$A', text_pos=(8e5, 1.2e4), rotation=90)
    add_box(ax, 250e3, 250e3*1.05, 100, 1.5e3, 'xkcd:green', r'T2K / Hyper-K', text_pos=(2.5e5, 9e3), rotation=90)
    
    # SBN experiments
    add_box(ax, 110, 110*1.05, 100, 2e3, 'xkcd:orange', r'SBND', text_pos=(110, 5e3), rotation=90)
    add_box(ax, 470, 470*1.05, 100, 2e3, 'xkcd:blue', r'MicroBooNE', text_pos=(600 * 0.6, 1e4), rotation=90)
    add_box(ax, 540, 540*1.05, 100, 2e3, 'xkcd:pink', r'MiniBooNE', text_pos=(600, 9e3), rotation=90)
    add_box(ax, 600, 600*1.05, 100, 2e3, 'xkcd:green', r'ICARUS', text_pos=(600 * 1.6, 6.5e3), rotation=90)
    
    # Short baseline experiments
    add_box(ax, 30, 30*1.05, 20, 60, 'xkcd:light blue', r'LSND', text_pos=(30, 150), rotation=90)
    add_box(ax, 24, 24*1.05, 10, 60, 'xkcd:lavender', r'JSNS$^2$', text_pos=(80, 30), rotation=90)
    add_box(ax, 48, 48*1.05, 10, 60, 'xkcd:lavender', r'')
    
    # Reactor experiments
    add_box(ax, 6.7, 9.2, 1.8, 10, 'xkcd:neon green', r'PROSPECT', text_pos=(8, 40), rotation=90)
    add_box(ax, 500, 500*1.05, 1.8, 10, 'xkcd:hot pink', r'Daya Bay', text_pos=(800, 15))
    add_box(ax, 1650, 1650*1.05, 1.8, 10, 'xkcd:hot pink', "")
    add_box(ax, 140e3, 210e3, 1.8, 10, 'xkcd:aqua', r'KamLAND', text_pos=(8e5, 7))
    add_box(ax, 52.5e3, 52.5e3*1.05, 1.8, 10, 'xkcd:neon purple', r'JUNO', text_pos=(2.5e4, 7))


L_over_E_max = 30000

# Physical parameters for neutrino oscillation
# Load parameters from nufit_params.py (normal ordering)
delta_m21_sq = nufit_params.delta2_m21  # Solar mass splitting (eV^2)
delta_m31_sq = nufit_params.delta2_m3l  # Atmospheric mass splitting (eV^2)

# Mixing angles (already in radians from nufit_params)
theta_12 = nufit_params.theta12  # Solar angle
theta_13 = nufit_params.theta13  # Reactor angle  
theta_23 = nufit_params.theta23  # Atmospheric angle

# CP violating phase (already in radians from nufit_params)
delta_CP = nufit_params.deltaCP

# Construct PMNS matrix using the function from nufit_params
# Set Majorana phases to 0 for simplicity
alpha1, alpha2 = 0.0, 0.0
U = nufit_params.get_PMNS(theta_12, theta_23, theta_13, delta_CP, alpha1, alpha2)

# Initial state: pure muon neutrino (flavor index 1)
initial_flavor = 1

# Colors for mass eigenstates
mass_colors = ['orange', 'purple', 'brown']

m1 = 0
m2 = np.sqrt(delta_m21_sq)
m3 = np.sqrt(delta_m31_sq)

# Oscillation frequencies (units of GeV/km)
omega_1 = 2 * 1.26693268 * m1**2
omega_2 = 2 * 1.26693268 * m2**2
omega_3 = 2 * 1.26693268 * m3**2

debug_print = False
if debug_print:
    print(f"PMNS Matrix:")
    print(f"U_e1={U[0,0]:.3f}, U_e2={U[0,1]:.3f}, U_e3={U[0,2]:.3f}")
    print(f"U_\\mu1={U[1,0]:.3f}, U_\\mu2={U[1,1]:.3f}, U_\\mu3={U[1,2]:.3f}")
    print(f"U_\\tau1={U[2,0]:.3f}, U_\\tau2={U[2,1]:.3f}, U_\\tau3={U[2,2]:.3f}")
    print(f"\nInitial coefficients (\\mu neutrino):")
    print(f"c_1={U[1,0]:.3f}, c_2={U[1,1]:.3f}, c_3={U[1,2]:.3f}")
    print(f"Sum |c_i|² = {np.sum(np.abs(U[1,:])**2):.3f}")

# Set up the figure with 4x2 subplots
fig, ((nue_vec_ax, osc_prob_ax), (numu_vec_ax, l_over_e_ax), (nutau_vec_ax, flavor_bar_ax)) = plt.subplots(3, 2, figsize=(14, 14))

# Setup vector plots for each flavor (left column)
vector_axes = [nue_vec_ax, numu_vec_ax, nutau_vec_ax]
flavor_names = ['Electron', 'Muon', 'Tau']
flavor_colors = ['tab:blue', 'tab:orange', 'tab:green']

for i, (ax, name, color) in enumerate(zip(vector_axes, flavor_names, flavor_colors)):
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{name} Neutrino Coefficient', fontsize=12)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    
    # Add unit circle for reference with flavor color
    circle = plt.Circle((0, 0), 1, fill=False, color=color, alpha=1, linestyle='--', lw=3)
    ax.add_patch(circle)

# Setup probability plot (top right) - will zoom during animation
osc_prob_ax.set_xlim(0, 100)  # Start with small range, will zoom during animation
osc_prob_ax.set_ylim(0, 1)
osc_prob_ax.set_xlabel('L/E (km/GeV)')
osc_prob_ax.set_ylabel('Probability')
osc_prob_ax.set_title('Oscillation Probabilities (Zooming)', fontsize=12)
#osc_prob_ax.grid(True, alpha=0.3)

# Setup L/E plot (bottom left)
l_over_e_ax.set_xscale('log')
l_over_e_ax.set_yscale('log')
l_over_e_ax.set_xlim(1, 1e8)  # L from 1 to 10^8 meters
l_over_e_ax.set_ylim(1, 1e6)  # E from 1 to 10^6 MeV
l_over_e_ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
l_over_e_ax.set_xlabel('L (meters)')
l_over_e_ax.set_ylabel('E (MeV)')
l_over_e_ax.grid(True, alpha=0.3)
l_over_e_ax.set_title('L vs E Relationship', fontsize=12)

# Setup flavor composition bar plot (bottom right)
flavor_bar_ax.set_xlim(0, 1)
flavor_bar_ax.set_ylim(0, 1)
flavor_bar_ax.set_xlabel('Flavor Composition')
flavor_bar_ax.set_ylabel('')
flavor_bar_ax.set_title('Neutrino Flavor Composition', fontsize=12)
flavor_bar_ax.set_xticks([])
flavor_bar_ax.set_yticks([])

# Call the function to set up the experiment plot
setup_experiment_plot(l_over_e_ax, delta_m21_sq, delta_m31_sq)

# Calculate theoretical oscillation probabilities over full range
L_over_E_plot = np.linspace(0, L_over_E_max*1.1, 1000)
P_ee_theory = np.zeros_like(L_over_E_plot*1.1)
P_mumu_theory = np.zeros_like(L_over_E_plot*1.1)
P_tautau_theory = np.zeros_like(L_over_E_plot*1.1)

for k, L_over_E in enumerate(L_over_E_plot):
    t = L_over_E  # Convert L/E to time
    # Time evolution operator
    phases = np.array([omega_1 * t, omega_2 * t, omega_3 * t])
    
    # Evolve the initial muon neutrino state
    # |ν(t)⟩ = Σᵢ U*ᵢμ e^(-iωᵢt) |νᵢ⟩
    # Amplitude for flavor α: ⟨να|ν(t)⟩ = Σᵢ U*ᵢμ e^(-iωᵢt) Uαᵢ
    
    amp_e = np.sum(np.conj(U[initial_flavor, :]) * np.exp(-1j * phases) * U[0, :])
    amp_mu = np.sum(np.conj(U[initial_flavor, :]) * np.exp(-1j * phases) * U[1, :])
    amp_tau = np.sum(np.conj(U[initial_flavor, :]) * np.exp(-1j * phases) * U[2, :])
    
    P_ee_theory[k] = np.abs(amp_e)**2
    P_mumu_theory[k] = np.abs(amp_mu)**2
    P_tautau_theory[k] = np.abs(amp_tau)**2

# Plot theoretical curves over full range (will be clipped by xlim during animation)
osc_prob_ax.plot(L_over_E_plot, P_ee_theory, 'b-', linewidth=2, alpha=0.7, label='$P(\\nu_\\mu \\to \\nu_e)$')
osc_prob_ax.plot(L_over_E_plot, P_mumu_theory, 'r-', linewidth=2, alpha=0.7, label='$P(\\nu_\\mu \\to \\nu_\\mu)$')
osc_prob_ax.plot(L_over_E_plot, P_tautau_theory, 'g-', linewidth=2, alpha=0.7, label='$P(\\nu_\\mu \\to \\nu_\\tau)$')

# Initialize animation objects
vectors = []
prob_points = []
prob_traces = []
l_e_line = None  # Will be initialized in animate function
flavor_bars = []  # Will store the bar plot rectangles

for i in range(3):
    ax = vector_axes[i]
    color = flavor_colors[i]
    
    # Individual mass eigenstate vectors
    vec_set = []
    for j in range(3):
        vec, = ax.plot([], [], 'o-', linewidth=2, markersize=6, 
                    alpha=0.7, label=f'$\\nu_{j+1}$', color=mass_colors[j])
        vec_set.append(vec)
    vectors.append(vec_set)
    
    ax.legend(loc='upper right', fontsize=8)
    
    # Probability point
    prob_point, = osc_prob_ax.plot([], [], 'o', markersize=8, color=color, alpha=0.8)
    prob_points.append(prob_point)
    
    # Probability trace
    prob_trace, = osc_prob_ax.plot([], [], '-', linewidth=2, color=color, alpha=0.8)
    prob_traces.append(prob_trace)

osc_prob_ax.legend(loc='upper right')

# Initialize flavor composition bars
bar_width = 0.3
bar_height = 0.1  # Take up middle 1/10 of plot
bar_y = 0.45  # Center vertically (0.5 - 0.1/2)

# Create three colored rectangles for the flavor composition bar
flavor_bar_colors = ['tab:blue', 'tab:orange', 'tab:green']
flavor_names = [r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$']

for i in range(3):
    bar = plt.Rectangle((0, bar_y), bar_width, bar_height, 
                        facecolor=flavor_bar_colors[i], alpha=0.7, edgecolor='black', linewidth=1)
    flavor_bar_ax.add_patch(bar)
    flavor_bars.append(bar)

# Add legend
legend_elements = []
for i, (color, name) in enumerate(zip(flavor_bar_colors, flavor_names)):
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, edgecolor='black'))
flavor_bar_ax.legend(legend_elements, flavor_names, loc='upper center', ncol=3, fontsize=10)

def animate(frame):
    # Initialize prev_l_e_line attribute if it doesn't exist
    if not hasattr(animate, 'prev_l_e_line'):
        animate.prev_l_e_line = None

    L_over_E_max = 40000
    L_over_E_start = 10

    log_L_over_E = (1 - frame / total_frames) * np.log(L_over_E_start) + frame / total_frames * np.log(L_over_E_max)
    L_over_E = np.exp(log_L_over_E)   

    # Update probability plot x-axis zoom
    max_x = L_over_E * 1.1
    if max_x == 0: 
        max_x = 1
    osc_prob_ax.set_xlim(0, max_x)
    
    # Calculate phases for each mass eigenstate
    phases = np.array([omega_1 * L_over_E, omega_2 * L_over_E, omega_3 * L_over_E])
    
    # For each flavor, calculate the amplitudes
    flavor_amplitudes = []
    
    for flavor_idx in range(3):
        # Calculate amplitude for this flavor
        # A_α(t) = Σᵢ U*ᵢμ e^(-iωᵢt) Uαᵢ
        amplitude = np.sum(np.conj(U[initial_flavor, :]) * np.exp(-1j * phases) * U[flavor_idx, :])
        flavor_amplitudes.append(amplitude)
        
        # Individual contributions from each mass eigenstate
        contributions = np.conj(U[initial_flavor, :]) * np.exp(-1j * phases) * U[flavor_idx, :]
        
        # Plot individual vectors tip-to-tail
        cumulative_pos = 0
        for i in range(3):
            start_pos = cumulative_pos
            end_pos = cumulative_pos + contributions[i]
            
            vectors[flavor_idx][i].set_data([start_pos.real, end_pos.real], 
                                        [start_pos.imag, end_pos.imag])
            
            cumulative_pos = end_pos
        
        # Update probability point
        prob = np.abs(amplitude)**2
        prob_points[flavor_idx].set_data([L_over_E], [prob])
        
        # Update probability trace
        if frame > 0:
            L_over_E_trace = np.linspace(0, L_over_E, min(frame, 200))
            prob_trace_vals = []
            for L_over_E_val in L_over_E_trace:
                t_val = L_over_E_val
                phases_trace = np.array([omega_1 * t_val, omega_2 * t_val, omega_3 * t_val])
                amp_trace = np.sum(np.conj(U[initial_flavor, :]) * np.exp(-1j * phases_trace) * U[flavor_idx, :])
                prob_trace_vals.append(np.abs(amp_trace)**2)
            prob_traces[flavor_idx].set_data(L_over_E_trace, prob_trace_vals)
    
    # Update flavor composition bar
    # Calculate current probabilities for each flavor
    probs = [np.abs(flavor_amplitudes[i])**2 for i in range(3)]
    total_prob = sum(probs)
    
    # Normalize to ensure they sum to 1
    if total_prob > 0:
        probs = [p / total_prob for p in probs]
    else:
        probs = [1/3, 1/3, 1/3]  # Equal distribution if all probabilities are zero
    
    # Update bar widths based on probabilities
    cumulative_width = 0
    for i in range(3):
        bar = flavor_bars[i]
        bar.set_width(probs[i])
        bar.set_x(cumulative_width)
        cumulative_width += probs[i]
    
    # Update L vs E line
    # For a given L/E ratio, we can plot a line showing all (L, E) combinations
    # that give this L/E ratio: E = L / (L/E)
    L_range = np.logspace(0, 8, 100)  # L from 1 to 10^8 meters
    
    # Avoid division by zero when L_over_E is 0
    if L_over_E > 0:
        E_values = L_range / L_over_E  # E = L / (L/E)
        
        # Only show points where E is within our plot range (1 to 10^6 MeV)
        valid_mask = (E_values >= 1) & (E_values <= 1e6)
        L_valid = L_range[valid_mask]
        E_valid = E_values[valid_mask]
        
        # Remove only the previous L/E line, not the entire plot
        if hasattr(animate, 'prev_l_e_line') and animate.prev_l_e_line is not None:
            animate.prev_l_e_line.remove()
        
        # Update title
        l_over_e_ax.set_title(f'L vs E Relationship (L/E = {L_over_E:.1f} km/GeV)', fontsize=12)
        
        if len(L_valid) > 0:
            l_e_line, = l_over_e_ax.plot(L_valid, E_valid, 'b-', linewidth=3, alpha=0.8)
            animate.prev_l_e_line = l_e_line
        else:
            l_e_line = None
            animate.prev_l_e_line = None
    else:
        # When L_over_E is 0, remove the line and update title
        if hasattr(animate, 'prev_l_e_line') and animate.prev_l_e_line is not None:
            animate.prev_l_e_line.remove()
            animate.prev_l_e_line = None
        l_over_e_ax.set_title('L vs E Relationship (L/E = 0 km/GeV)', fontsize=12)
        l_e_line = None
    
    # Return all animated objects
    all_objects = []
    for vec_set in vectors:
        all_objects.extend(vec_set)
    all_objects.extend(prob_points)
    all_objects.extend(prob_traces)
    all_objects.extend(flavor_bars)
    if l_e_line is not None:
        all_objects.append(l_e_line)
    
    return all_objects

# Create animation
total_frames = 200  # Reasonable number of frames for GIF
anim = FuncAnimation(fig, animate, frames=total_frames, interval=100, blit=True)

plt.tight_layout()

# Save the animation as a GIF with progress bar
print(f"Saving animation to plots/neutrino_oscillation_{L_over_E_max}.gif")

# Create a progress bar for the save operation
pbar = tqdm(total=total_frames, desc="Rendering animation")

# Override the grab_frame method to show progress
original_grab_frame = anim._func
def animate_with_progress(frame):
    pbar.update(1)
    return original_grab_frame(frame)

anim._func = animate_with_progress
anim.save(f'plots/neutrino_oscillation.gif', writer='pillow', fps=20)
pbar.close()

