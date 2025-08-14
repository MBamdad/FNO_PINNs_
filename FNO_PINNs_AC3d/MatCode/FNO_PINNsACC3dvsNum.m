%% MATLAB Script for Comparing 3D Allen-Cahn Numerical vs FNO-PINN Results
% Version: Includes interpolation for different grid sizes and all helpers.

clc;
clearvars;
close all;
fclose('all');
disp('START: Allen-Cahn 3D Comparison Script')

%% ========= Section 1: Run Numerical Simulation =========
disp('--- Running Numerical Simulation ---');
% --- Physical Parameters ---
lambda_param = 10.0; epsilon_param = 0.025; domain_bounds = [0.0, 1.0];
time_bounds = [0.0, 5.0]; drop_radius_param = 0.35; drop_center_x = 0.5;
drop_center_y = 0.5; drop_center_z = 0.5;
% --- Spatial Parameters ---
Nx = 64; Ny = Nx; Nz = Nx; Lx = domain_bounds(2) - domain_bounds(1); Ly = Lx; Lz = Lx;
hx = Lx / Nx; hy = Ly / Ny; hz = Lz / Nz;
x_num_vec = linspace(domain_bounds(1), domain_bounds(2), Nx);
y_num_vec = linspace(domain_bounds(1), domain_bounds(2), Ny);
z_num_vec = linspace(domain_bounds(1), domain_bounds(2), Nz);
[X_nd, Y_nd, Z_nd] = ndgrid(x_num_vec, y_num_vec, z_num_vec); % For simulation
% --- DFT Setup ---
p = (2*pi/Lx) * [0:Nx/2, -Nx/2+1:-1]; q = (2*pi/Ly) * [0:Ny/2, -Ny/2+1:-1]; r = (2*pi/Lz) * [0:Nz/2, -Nz/2+1:-1];
p2 = p.^2; q2 = q.^2; r2 = r.^2; [pp2, qq2, rr2] = ndgrid(p2, q2, r2); laplacian_k = -(pp2 + qq2 + rr2);
% --- Time Discretization ---
T_final = time_bounds(2); dt = 5e-3; Nt = round(T_final / dt); T_actual = Nt * dt;
fprintf('Numerical Sim: Running until T = %.4f with dt = %.2e (%d steps)\n', T_actual, dt, Nt);
% --- Specify times to save ---
target_save_times = [0.0, 1.5, 2.5, 5.0]; % Match PINN times
num_target_steps = length(target_save_times); save_indices = zeros(1, num_target_steps);
for i = 1:num_target_steps; save_indices(i) = round(target_save_times(i) / dt) + 1; end
save_indices(save_indices < 1) = 1; save_indices(save_indices > Nt + 1) = Nt + 1; save_indices = unique(save_indices);
actual_num_saved_num = length(save_indices); saved_times_num = (save_indices - 1) * dt;
fprintf('Numerical solver will save %d snapshots at times approx: %s\n', actual_num_saved_num, mat2str(saved_times_num, 3));
results_u_num = zeros(Nx, Ny, Nz, actual_num_saved_num);
% --- Initial Condition ---
disp('Setting numerical initial condition...');
distance = sqrt((X_nd - drop_center_x).^2 + (Y_nd - drop_center_y).^2 + (Z_nd - drop_center_z).^2);
u = tanh((drop_radius_param - distance) / (2 * epsilon_param));
disp('Initial condition set.');
% --- Time Stepping ---
disp('Starting numerical time evolution...');
save_counter_num = 1;
if ~isempty(save_indices) && save_indices(save_counter_num) == 1
    results_u_num(:, :, :, save_counter_num) = u;
    fprintf('Numerical: Saved snapshot at t = %.4f (Iter 1)\n', saved_times_num(save_counter_num));
    save_counter_num = save_counter_num + 1;
end
denominator_factor = (1.0/dt - lambda_param * epsilon_param^2 * laplacian_k);
denominator_factor(abs(denominator_factor) < 1e-12) = 1e-12;
tic;
for iter = 1:Nt
    u_real = real(u);
    nonlinear_term_Fprime = u_real.^3 - u_real;
    u_hat = fftn(u_real);
    nonlinear_term_Fprime_hat = fftn(nonlinear_term_Fprime);
    numerator_factor = u_hat / dt - lambda_param * nonlinear_term_Fprime_hat;
    u_hat_new = numerator_factor ./ denominator_factor;
    u = ifftn(u_hat_new);
    if any(isnan(u(:))) || any(isinf(u(:))); error('Numerical simulation unstable at iteration %d (t=%.4f).', iter, iter*dt); end
    if save_counter_num <= actual_num_saved_num && (iter + 1) == save_indices(save_counter_num)
        results_u_num(:, :, :, save_counter_num) = real(u);
        fprintf('Numerical: Saved snapshot at t = %.4f (Iter %d)\n', saved_times_num(save_counter_num), iter + 1);
        save_counter_num = save_counter_num + 1;
    end
end
sim_time = toc;
disp(['Numerical time evolution finished in ', num2str(sim_time), ' seconds.']);
disp('--- Numerical Simulation Finished ---');

%% ========= Section 2: Load FNO-PINN Results =========
disp('--- Loading FNO-PINN Results ---');
pinn_filename = '/scratch/noqu8762/PhaseField_SANO3D/PF_PINNs/ac3d_final_results_100Epochs.mat'; % Assumes file is in the same directory
if ~exist(pinn_filename, 'file'); error('FNO-PINN results file not found: %s', pinn_filename); end

fprintf('Loading FNO-PINN data from: %s\n', pinn_filename);
pinn_data = load(pinn_filename);

% --- Data Validation ---
required_vars = {'T', 'U'};
if ~all(isfield(pinn_data, required_vars)); error('Loaded .mat file is missing required variables (T, U).'); end

pinn_T = pinn_data.T(:)';
pinn_U = pinn_data.U;

% --- Create FNO-PINN grid vectors ---
pinn_Nx = size(pinn_U, 1);
pinn_Ny = size(pinn_U, 2);
pinn_Nz = size(pinn_U, 3);
pinn_x_vec = linspace(domain_bounds(1), domain_bounds(2), pinn_Nx);
pinn_y_vec = linspace(domain_bounds(1), domain_bounds(2), pinn_Ny);
pinn_z_vec = linspace(domain_bounds(1), domain_bounds(2), pinn_Nz);

fprintf('FNO-PINN data loaded successfully.\n');
fprintf('FNO-PINN Times (T): %s\n', mat2str(pinn_T, 3));
fprintf('FNO-PINN Solution size (U): [%d %d %d %d]\n', size(pinn_U,1), size(pinn_U,2), size(pinn_U,3), size(pinn_U,4));

if pinn_Nx ~= Nx || pinn_Ny ~= Ny || pinn_Nz ~= Nz
    warning('FNO-PINN grid (%d^3) does not match numerical grid (%d^3). Interpolation will be used.', pinn_Nx, Nx);
end


%% ========= Section 3: Find Common Times, Interpolate, and Calculate Errors =========
disp('--- Aligning Time Points, Interpolating, and Calculating Errors ---');

time_tolerance = dt / 2;
[common_times, num_indices, pinn_indices] = intersect_detailed(saved_times_num, pinn_T, time_tolerance);
num_comparison_times = length(common_times);

if num_comparison_times == 0
    error('No common time points found between numerical and FNO-PINN results.');
else
    fprintf('Found %d common time points for comparison:\n', num_comparison_times);
    disp(common_times);
end

% Create meshgrids for interpolation
[pinn_X_nd, pinn_Y_nd, pinn_Z_nd] = ndgrid(pinn_x_vec, pinn_y_vec, pinn_z_vec);
[num_X_nd_target, num_Y_nd_target, num_Z_nd_target] = ndgrid(x_num_vec, y_num_vec, z_num_vec); % This is the target 64^3 grid

relative_l2_errors = zeros(1, num_comparison_times);
signed_error_fields = cell(1, num_comparison_times);

for k = 1:num_comparison_times
    U_num_k = results_u_num(:, :, :, num_indices(k));
    U_pinn_low_res = pinn_U(:, :, :, pinn_indices(k));

    % --- INTERPOLATE FNO-PINN DATA TO NUMERICAL GRID ---
    fprintf('Interpolating FNO-PINN solution at t=%.3f from [%d^3] to [%d^3]...\n', ...
            common_times(k), pinn_Nx, Nx);
    U_pinn_k = interpn(pinn_X_nd, pinn_Y_nd, pinn_Z_nd, U_pinn_low_res, ...
                       num_X_nd_target, num_Y_nd_target, num_Z_nd_target, 'linear');

    error_field_k = real(U_num_k) - real(U_pinn_k);
    signed_error_fields{k} = error_field_k;

    diff_norm = norm(error_field_k(:));
    norm_ref = norm(real(U_num_k(:)));

    if norm_ref < eps; relative_l2_errors(k) = diff_norm; else; relative_l2_errors(k) = diff_norm / norm_ref; end
    fprintf('t = %.3f: Relative L2 Error = %.4e\n', common_times(k), relative_l2_errors(k));
end

%% ========= Section 4: Create Comparison Plots =========
disp('--- Generating Comparison Plots ---');

figure('Name', '3D Allen-Cahn: Numerical vs FNO-PINN Comparison', 'Position', [50, 50, 350*num_comparison_times, 900]);
sgtitle('3D Allen-Cahn: Numerical vs FNO-PINN', 'FontSize', 14, 'FontWeight', 'bold');

[X_plot_mg, Y_plot_mg, Z_plot_mg] = meshgrid(x_num_vec, y_num_vec, z_num_vec);
isovalue = 0;
domain_axis_limits = [domain_bounds(1) domain_bounds(2) domain_bounds(1) domain_bounds(2) domain_bounds(1) domain_bounds(2)];
plot_view = [-45 45];
plot_title_fontsize = 10;
plot_label_fontsize = 9;
plot_tick_fontsize = 8;
slice_index = round(Nz/2);

for k = 1:num_comparison_times
    current_t = common_times(k);

    % --- Row 1: Numerical Result (Isosurface) ---
    subplot(3, num_comparison_times, k);
    U_num_mg = permute(real(results_u_num(:, :, :, num_indices(k))), [2 1 3]);
    plot_isosurface_mod(X_plot_mg, Y_plot_mg, Z_plot_mg, U_num_mg, isovalue, sprintf('Numerical t=%.2f', current_t), [0.2, 0.7, 1.0], domain_axis_limits, plot_view, plot_title_fontsize, plot_label_fontsize, plot_tick_fontsize);

    % --- Row 2: FNO-PINN Result (Isosurface) ---
    subplot(3, num_comparison_times, k + num_comparison_times);
    U_pinn_interp_nd = interpn(pinn_X_nd, pinn_Y_nd, pinn_Z_nd, pinn_U(:, :, :, pinn_indices(k)), num_X_nd_target, num_Y_nd_target, num_Z_nd_target, 'linear');
    U_pinn_mg = permute(real(U_pinn_interp_nd), [2 1 3]);
    plot_isosurface_mod(X_plot_mg, Y_plot_mg, Z_plot_mg, U_pinn_mg, isovalue, sprintf('FNO-PINN t=%.2f', current_t), [1.0, 0.5, 0.2], domain_axis_limits, plot_view, plot_title_fontsize, plot_label_fontsize, plot_tick_fontsize);

    % --- Row 3: SIGNED Pointwise Error Slice ---
    subplot(3, num_comparison_times, k + 2*num_comparison_times);
    error_slice = squeeze(signed_error_fields{k}(:, :, slice_index))';
    contourf(x_num_vec, y_num_vec, error_slice, 50, 'LineStyle', 'none');
    axis equal tight; box on; grid on;
    colormap(gca, 'jet');
    max_abs_err = max(abs(error_slice(:)));
    if max_abs_err > 1e-9; caxis([-max_abs_err, max_abs_err]); else; caxis([-1e-9, 1e-9]); end
    colorbar;
    title(sprintf('Error @ z=%.2f (L2=%.2e)', z_num_vec(slice_index), relative_l2_errors(k)), 'FontSize', plot_title_fontsize);
    xlabel('x', 'FontSize', plot_label_fontsize); ylabel('y', 'FontSize', plot_label_fontsize);
end
disp('--- Comparison Script Finished ---');


%% ========= HELPER FUNCTION: Plot Isosurface =========
function plot_isosurface_mod(X_mg, Y_mg, Z_mg, U_mg, isovalue, title_str, face_color, axis_limits, view_angle, title_fs, label_fs, tick_fs)
    cla;
    fv = isosurface(X_mg, Y_mg, Z_mg, U_mg, isovalue);
    if isempty(fv.vertices)
         title({title_str, '(No surface)'}, 'FontSize', title_fs);
    else
        p = patch(fv);
        isonormals(X_mg, Y_mg, Z_mg, U_mg, p);
        p.FaceColor = face_color; p.EdgeColor = 'none'; p.FaceLighting = 'gouraud';
        p.AmbientStrength = 0.4; p.DiffuseStrength = 0.8; p.SpecularStrength = 0.2;
        title(title_str, 'FontSize', title_fs);
    end
    axis(axis_limits); view(view_angle); grid on; box on; daspect([1 1 1]);
    camlight headlight; camlight right; lighting gouraud;
    xlabel('x', 'FontSize', label_fs); ylabel('y', 'FontSize', label_fs); zlabel('z', 'FontSize', label_fs);
    set(gca, 'FontSize', tick_fs);
end

%% ========= HELPER FUNCTION: Intersect with Tolerance =========
function [common_elements, idxA, idxB] = intersect_detailed(A, B, tol)
    if nargin < 3; tol = 1e-9; end
    A = A(:); B = B(:);
    [~, sortA] = sort(A); [~, sortB] = sort(B);
    [~, unsortA] = sort(sortA); [~, unsortB] = sort(sortB);
    A_sorted = A(sortA); B_sorted = B(sortB);

    i = 1; j = 1;
    idxA_sorted = []; idxB_sorted = [];
    while i <= length(A_sorted) && j <= length(B_sorted)
        if abs(A_sorted(i) - B_sorted(j)) <= tol
            idxA_sorted = [idxA_sorted; i];
            idxB_sorted = [idxB_sorted; j];
            i = i + 1; j = j + 1;
        elseif A_sorted(i) < B_sorted(j)
            i = i + 1;
        else
            j = j + 1;
        end
    end

    idxA = sortA(idxA_sorted);
    idxB = sortB(idxB_sorted);
    common_elements = A(idxA);
end