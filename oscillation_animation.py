import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
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
    add_box(ax, 1.3e6, 1.3e6*1.05, 750, 5e3, 'xkcd:red', r'DUNE', text_pos=(1.3e6, 1.3e4), edgecolor='xkcd:red', rotation=90)
    add_box(ax, 810e3, 810e3*1.04, 500, 5e3, 'xkcd:light orange', r'NO$\nu$A', text_pos=(8e5, 1.2e4), rotation=90)
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


def create_neutrino_oscillation_animation(n_flavors=3, L_over_E_max=None, save_video=True):
    """
    Create neutrino oscillation animation.
    
    Parameters:
    n_flavors (int): Number of neutrino flavors (2, 3, or 4)
                    2: Simple 2-flavor oscillation (electron-muon)
                    3: Standard 3-flavor oscillation
                    4: 4-flavor with sterile neutrino
    L_over_E_max (float): Maximum L/E ratio for animation. If None, uses default values
    save_video (bool): Whether to save animation as MP4 video
    """
    
    # Set default L_over_E_max based on number of flavors
    if L_over_E_max is None:
        if n_flavors == 2:
            L_over_E_max = 2000  # Shorter range for 2-flavor
        else:
            L_over_E_max = 40000  # Standard 3-flavor range
    
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
    
    # Setup parameters based on number of flavors
    if n_flavors == 2:
        # Simple 2-flavor oscillation (electron-muon with atmospheric parameters)
        print("=== 2-FLAVOR MODE ===")
        
        # Use atmospheric parameters for more visible oscillation
        theta = theta_23  # Use atmospheric mixing angle
        delta_m_sq = delta_m31_sq  # Use atmospheric mass splitting
        
        # Construct 2x2 PMNS matrix
        U = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        
        # Number of flavors and masses
        masses = [0, np.sqrt(delta_m_sq)]
        flavor_names = ['Flavor A', 'Flavor B']
        flavor_colors = ['tab:blue', 'tab:orange']
        mass_colors = ['tab:green', 'tab:red']
        flavor_labels = [r'$\nu_a$', r'$\nu_b$']
        prob_labels = [
            '$P(\\nu_b \\to \\nu_a)$',
            '$P(\\nu_b \\to \\nu_b)$'
        ]
        
    elif n_flavors == 4:
        # LSND best-fit point parameters for sterile neutrino
        sin2_2thetamue = 0.003
        delta_m2_14 = 1.2
        m4 = np.sqrt(delta_m2_14)
        
        # Calculate sterile mixing angles
        sin2_2theta14 = 0.36  # matching a test point
        sin2_theta24 = sin2_2thetamue / sin2_2theta14
        theta14 = np.arcsin(np.sqrt(sin2_2theta14)) / 2
        theta24 = np.arcsin(np.sqrt(sin2_theta24))
        theta34 = 0  # set theta34 to 0
        delta24 = 0
        delta34 = 0
        
        print("=== STERILE NEUTRINO MODE ===")
        print(f"sin2_2theta14 = {np.sin(2 * theta14)**2:.6f}")
        print(f"sin2_theta24 = {np.sin(theta24)**2:.6f}")
        print(f"theta14 = {theta14 * 180 / np.pi:.3f} deg")
        print(f"theta24 = {theta24 * 180 / np.pi:.3f} deg")
        print(f"sin2_2thetamue = {sin2_2thetamue:.6f}")
        
        # Get 4x4 PMNS matrix
        U = nufit_params.get_sterile_PMNS(theta_12, theta_23, theta_13, delta_CP, 
                                         theta14, theta24, theta34, delta24, delta34)
        
        # Number of flavors and masses
        masses = [0, np.sqrt(delta_m21_sq), np.sqrt(delta_m31_sq), m4]
        flavor_names = ['Electron', 'Muon', 'Tau', 'Sterile']
        flavor_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        mass_colors = ['tab:purple', 'tab:cyan', 'tab:pink', 'tab:olive']
        flavor_labels = [r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$', r'$\nu_s$']
        prob_labels = [
            '$P(\\nu_\\mu \\to \\nu_e)$',
            '$P(\\nu_\\mu \\to \\nu_\\mu)$',
            '$P(\\nu_\\mu \\to \\nu_\\tau)$',
            '$P(\\nu_\\mu \\to \\nu_s)$'
        ]
        
    else:
        # Standard 3-flavor neutrino oscillations
        # Construct PMNS matrix using the function from nufit_params
        # Set Majorana phases to 0 for simplicity
        alpha1, alpha2 = 0.0, 0.0
        U = nufit_params.get_PMNS(theta_12, theta_23, theta_13, delta_CP, alpha1, alpha2)
        
        print("=== STANDARD 3-FLAVOR MODE ===")
        
        # Number of flavors and masses
        masses = [0, np.sqrt(delta_m21_sq), np.sqrt(delta_m31_sq)]
        flavor_names = ['Electron', 'Muon', 'Tau']
        flavor_colors = ['tab:blue', 'tab:orange', 'tab:green']
        mass_colors = ['tab:purple', 'tab:cyan', 'tab:pink']
        flavor_labels = [r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$']
        prob_labels = [
            '$P(\\nu_\\mu \\to \\nu_e)$',
            '$P(\\nu_\\mu \\to \\nu_\\mu)$',
            '$P(\\nu_\\mu \\to \\nu_\\tau)$'
        ]
    
    # Initial state: pure muon neutrino (flavor index 1)
    initial_flavor = 1
    
    # Oscillation frequencies (units of GeV/km)
    omegas = [2 * 1.26693268 * m**2 for m in masses]
    
    # Set up the figure with appropriate subplot layout
    if n_flavors == 2:
        # 2x2 layout for 2-flavor: left column has vector plots, right column has prob and bar
        fig, ((nue_vec_ax, osc_prob_ax), (numu_vec_ax, flavor_bar_ax)) = plt.subplots(2, 2, figsize=(12, 10))
        vector_axes = [nue_vec_ax, numu_vec_ax]
        l_over_e_ax = None  # No L/E plot for 2-flavor
    elif n_flavors == 4:
        fig, ((nue_vec_ax, osc_prob_ax), (numu_vec_ax, l_over_e_ax), 
              (nutau_vec_ax, flavor_bar_ax), (nus_vec_ax, unused_ax)) = plt.subplots(4, 2, figsize=(14, 16))
        vector_axes = [nue_vec_ax, numu_vec_ax, nutau_vec_ax, nus_vec_ax]
        unused_ax.axis('off')  # Hide the unused subplot
    else:
        fig, ((nue_vec_ax, osc_prob_ax), (numu_vec_ax, l_over_e_ax), 
              (nutau_vec_ax, flavor_bar_ax)) = plt.subplots(3, 2, figsize=(14, 12))
        vector_axes = [nue_vec_ax, numu_vec_ax, nutau_vec_ax]
    
    # Setup vector plots for each flavor (left column)
    for i, (ax, name, color) in enumerate(zip(vector_axes, flavor_names, flavor_colors)):
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'{name} Neutrino Coefficient', fontsize=12)
        #ax.set_xlabel('Real Part')
        #ax.set_ylabel('Imaginary Part')
        
        # Add unit circle for reference with flavor color
        circle = plt.Circle((0, 0), 1, fill=False, color=color, alpha=1, linestyle='--', lw=3)
        ax.add_patch(circle)

        # Remove ticks and grid to match PMNS style
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove subplot borders to match PMNS style
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Setup probability plot (top right) - will zoom during animation
    osc_prob_ax.set_xlim(0, 100)  # Start with small range, will zoom during animation
    osc_prob_ax.set_ylim(0, 1)
    osc_prob_ax.set_xlabel('L/E (km/GeV)')
    osc_prob_ax.set_ylabel('Probability')
    osc_prob_ax.set_title('Oscillation Probabilities', fontsize=12)
    
    # Setup L/E plot (only for 3 and 4 flavor modes)
    if l_over_e_ax is not None:
        l_over_e_ax.set_xscale('log')
        l_over_e_ax.set_yscale('log')
        l_over_e_ax.set_xlim(1, 1e8)  # L from 1 to 10^8 meters
        l_over_e_ax.set_ylim(1, 1e6)  # E from 1 to 10^6 MeV
        l_over_e_ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        l_over_e_ax.set_xlabel('L (meters)')
        l_over_e_ax.set_ylabel('E (MeV)')
        l_over_e_ax.grid(True, alpha=0.3)
        l_over_e_ax.set_title('L vs E Relationship', fontsize=12)
        
        # Call the function to set up the experiment plot
        setup_experiment_plot(l_over_e_ax, delta_m21_sq, delta_m31_sq)
    
    # Setup flavor composition bar plot
    flavor_bar_ax.set_xlim(0, 1)
    flavor_bar_ax.set_ylim(0, 1)
    flavor_bar_ax.set_xlabel('Flavor Composition')
    flavor_bar_ax.set_ylabel('')
    flavor_bar_ax.set_title('Neutrino Flavor Composition', fontsize=12)
    flavor_bar_ax.set_xticks([])
    flavor_bar_ax.set_yticks([])
    
        # Initialize animation objects
    vectors = []
    prob_points = []
    prob_traces = []
    flavor_bars = []
    
    for i in range(n_flavors):
        ax = vector_axes[i]
        color = flavor_colors[i]
        
        # Individual mass eigenstate vectors
        vec_set = []
        for j in range(n_flavors):
            color = mass_colors[j]
            # We'll create arrows each frame to match PMNS style exactly
            vec_set.append(None)  # Placeholder for arrow objects
        vectors.append(vec_set)
        
        # Add dummy plots for legend entries
        for j in range(n_flavors):
            color = mass_colors[j]
            # Create invisible dummy plot for legend
            ax.plot([], [], color=color, linewidth=3, label=f'$\\nu_{j+1}$')
        ax.legend(loc='upper right', fontsize=8)
        
        # Probability point
        prob_point, = osc_prob_ax.plot([], [], 'o', markersize=8, color=flavor_colors[i])
        prob_points.append(prob_point)
        
        # Probability trace
        prob_trace, = osc_prob_ax.plot([], [], '-', linewidth=2, color=flavor_colors[i], label=prob_labels[i])
        prob_traces.append(prob_trace)
    
    osc_prob_ax.legend(loc='upper right')
    
    # Initialize flavor composition bars
    bar_height = 0.1  # Take up middle 1/10 of plot
    bar_y = 0.45  # Center vertically (0.5 - 0.1/2)
    
    for i in range(n_flavors):
        bar = plt.Rectangle((0, bar_y), 0, bar_height, facecolor=flavor_colors[i], 
                           alpha=0.7, edgecolor='black', linewidth=1)
        flavor_bar_ax.add_patch(bar)
        flavor_bars.append(bar)
    
    # Add legend
    legend_elements = []
    for i, (color, name) in enumerate(zip(flavor_colors, flavor_labels)):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, edgecolor='black'))
    flavor_bar_ax.legend(legend_elements, flavor_labels, loc='upper center', 
                        ncol=n_flavors, fontsize=10)
    
    def animate(frame):
        # Initialize prev_l_e_line attribute if it doesn't exist
        if not hasattr(animate, 'prev_l_e_line'):
            animate.prev_l_e_line = None
        
        L_over_E_start = 0

        if n_flavors == 3:
            # Linear until L/E=1000, then exponential
            transition_L_over_E = 1000
            transition_L_over_E_2 = 5000
            transition_frame = 0.4 * total_frames
            transition_frame_2 = 0.7 * total_frames
            if frame <= transition_frame:
                L_over_E = L_over_E_start + (transition_L_over_E - L_over_E_start) * frame / transition_frame
            elif frame <= transition_frame_2:
                L_over_E = transition_L_over_E + (transition_L_over_E_2 - transition_L_over_E) * (frame - transition_frame) / (transition_frame_2 - transition_frame)
            else:
                L_over_E = transition_L_over_E_2 + (L_over_E_max - transition_L_over_E_2) * (frame - transition_frame_2) / (total_frames - transition_frame_2)
        elif n_flavors == 4:
            transition_L_over_E = 2
            transition_L_over_E_2 = 10
            transition_L_over_E_3 = 100
            transition_L_over_E_4 = 1000
            transition_L_over_E_5 = 5000
            transition_frame = 0.25 * total_frames
            transition_frame_2 = 0.5 * total_frames
            transition_frame_3 = 0.625 * total_frames
            transition_frame_4 = 0.75 * total_frames
            transition_frame_5 = 0.875 * total_frames
            if frame <= transition_frame:
                L_over_E = L_over_E_start + (transition_L_over_E - L_over_E_start) * frame / transition_frame
            elif frame <= transition_frame_2:
                L_over_E = transition_L_over_E + (transition_L_over_E_2 - transition_L_over_E) * (frame - transition_frame) / (transition_frame_2 - transition_frame)
            elif frame <= transition_frame_3:
                L_over_E = transition_L_over_E_2 + (transition_L_over_E_3 - transition_L_over_E_2) * (frame - transition_frame_2) / (transition_frame_3 - transition_frame_2)
            elif frame <= transition_frame_4:
                L_over_E = transition_L_over_E_3 + (transition_L_over_E_4 - transition_L_over_E_3) * (frame - transition_frame_3) / (transition_frame_4 - transition_frame_3)
            elif frame <= transition_frame_5:
                L_over_E = transition_L_over_E_4 + (transition_L_over_E_5 - transition_L_over_E_4) * (frame - transition_frame_4) / (transition_frame_5 - transition_frame_4)
            else:
                L_over_E = transition_L_over_E_5 + (L_over_E_max - transition_L_over_E_5) * (frame - transition_frame_5) / (total_frames - transition_frame_5)
        else:
            L_over_E = (1 - frame / total_frames) * L_over_E_start + frame / total_frames * L_over_E_max
        
        # Update probability plot x-axis zoom
        max_x = L_over_E * 1.1
        if max_x == 0: 
            max_x = 1
        osc_prob_ax.set_xlim(0, max_x)
        
        if n_flavors == 4:
            if frame <= transition_frame_2:
                osc_prob_ax.set_ylim(0, 0.05)
            else:
                frac_done_transition_2_3 = min(1, (frame - transition_frame_2) / (transition_frame_3 - transition_frame_2))
                osc_prob_ax.set_ylim(0, (1 - frac_done_transition_2_3) * 0.05 + frac_done_transition_2_3 * 1)
        
        # Calculate phases for each mass eigenstate
        phases = np.array([omega * L_over_E for omega in omegas])
        
        # For each flavor, calculate the amplitudes
        flavor_amplitudes = []
        
        for flavor_idx in range(n_flavors):
            # Calculate amplitude for this flavor
            # A_α(t) = Σᵢ U*ᵢμ e^(-iωᵢt) Uαᵢ
            amplitude = np.sum(np.conj(U[initial_flavor, :]) * np.exp(-1j * phases) * U[flavor_idx, :])
            flavor_amplitudes.append(amplitude)
            
            # Individual contributions from each mass eigenstate
            contributions = np.conj(U[initial_flavor, :]) * np.exp(-1j * phases) * U[flavor_idx, :]
            
            # Plot individual vectors tip-to-tail
            cumulative_pos = 0
            ax = vector_axes[flavor_idx]
            
            # Clear previous arrows for this flavor
            if hasattr(animate, f'prev_arrows_{flavor_idx}'):
                for arrow in getattr(animate, f'prev_arrows_{flavor_idx}'):
                    if arrow is not None:
                        arrow.remove()
            
            current_arrows = []
            for i in range(n_flavors):
                start_pos = cumulative_pos
                end_pos = cumulative_pos + contributions[i]
                
                # Create new arrow matching PMNS style exactly
                if abs(end_pos - start_pos) > 1e-10:  # Only draw if arrow has non-zero length
                    arrow_color = mass_colors[i]
                    
                    # Adjust end position to match PMNS treatment (shorten arrow by 0.1)
                    arrow_vector = end_pos - start_pos
                    arrow_magnitude = abs(arrow_vector)
                    if arrow_magnitude > 0.1:  # Only adjust if arrow is long enough
                        adjusted_magnitude = arrow_magnitude - 0.1
                        adjusted_end_pos = start_pos + (arrow_vector / arrow_magnitude) * adjusted_magnitude
                    else:
                        adjusted_end_pos = end_pos
                    
                    arrow = ax.arrow(start_pos.real, start_pos.imag, 
                                   (adjusted_end_pos - start_pos).real, (adjusted_end_pos - start_pos).imag,
                                   head_width=0.08, head_length=0.08, 
                                   fc=arrow_color, ec=arrow_color, linewidth=3)
                    current_arrows.append(arrow)
                else:
                    current_arrows.append(None)
                
                cumulative_pos = end_pos
            
            # Store current arrows for next frame cleanup
            setattr(animate, f'prev_arrows_{flavor_idx}', current_arrows)
            
            # Update probability point
            prob = np.abs(amplitude)**2
            prob_points[flavor_idx].set_data([L_over_E], [prob])
            
            # Update probability trace
            if frame > 0:
                # Use fixed number of points for smooth curve, independent of frame count
                n_points = 10000  # Increased number of points for smoother curve
                L_over_E_trace = np.linspace(0, L_over_E, n_points)
                
                # Vectorized calculation instead of loop for efficiency
                phases_trace = np.outer(L_over_E_trace, omegas)  # Shape: (n_points, n_flavors)
                amp_trace = np.sum(np.conj(U[initial_flavor, :]) * np.exp(-1j * phases_trace) * U[flavor_idx, :], axis=1)
                prob_trace_vals = np.abs(amp_trace)**2
                
                prob_traces[flavor_idx].set_data(L_over_E_trace, prob_trace_vals)
        
        # Update flavor composition bar
        # Calculate current probabilities for each flavor
        probs = [np.abs(flavor_amplitudes[i])**2 for i in range(n_flavors)]
        total_prob = sum(probs)
        
        # Normalize to ensure they sum to 1
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1/n_flavors] * n_flavors  # Equal distribution if all probabilities are zero
        
        # Update bar widths based on probabilities
        cumulative_width = 0
        for i in range(n_flavors):
            bar = flavor_bars[i]
            bar.set_width(probs[i])
            bar.set_x(cumulative_width)
            cumulative_width += probs[i]
        
        # Update L vs E line (only if L/E plot exists)
        l_e_line = None
        if l_over_e_ax is not None:
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
        # Add current arrows for this frame
        for flavor_idx in range(n_flavors):
            if hasattr(animate, f'prev_arrows_{flavor_idx}'):
                current_arrows = getattr(animate, f'prev_arrows_{flavor_idx}')
                for arrow in current_arrows:
                    if arrow is not None:
                        all_objects.append(arrow)
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
    
    if save_video:
        # Save the animation as an MP4 with progress bar
        if n_flavors == 2:
            filename_suffix = "2flavor"
        elif n_flavors == 4:
            filename_suffix = "sterile"
        else:
            filename_suffix = "standard"
        
        mp4_filename = f'plots/neutrino_oscillation_{filename_suffix}.mp4'
        print(f"Saving animation to {mp4_filename}")
        
        # Create a progress bar for the save operation
        pbar = tqdm(total=total_frames, desc="Rendering animation")
        
        # Override the grab_frame method to show progress
        original_grab_frame = anim._func
        def animate_with_progress(frame):
            pbar.update(1)
            return original_grab_frame(frame)
        
        anim._func = animate_with_progress
        anim.save(mp4_filename, writer='ffmpeg', fps=20)
        pbar.close()
    
    return fig, anim


# Convenience functions for specific oscillation types
def create_2flavor_animation(L_over_E_max=None, save_video=True):
    """Create simple 2-flavor neutrino oscillation animation."""
    return create_neutrino_oscillation_animation(n_flavors=2, L_over_E_max=L_over_E_max, save_video=save_video)

def create_3flavor_animation(L_over_E_max=None, save_video=True):
    """Create standard 3-flavor neutrino oscillation animation."""
    return create_neutrino_oscillation_animation(n_flavors=3, L_over_E_max=L_over_E_max, save_video=save_video)

def create_sterile_animation(L_over_E_max=None, save_video=True):
    """Create 4-flavor neutrino oscillation animation with sterile neutrino."""
    return create_neutrino_oscillation_animation(n_flavors=4, L_over_E_max=L_over_E_max, save_video=save_video)


# Main execution
if __name__ == "__main__":
    # Create 2-flavor animation (simple case)
    fig1, anim1 = create_2flavor_animation()
    
    # Create standard 3-flavor animation
    fig2, anim2 = create_3flavor_animation()
    
    # Create sterile neutrino animation
    fig3, anim3 = create_sterile_animation()
    
    print("done!")

