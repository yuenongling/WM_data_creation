% How are dimensionless  numbers defined
% Cf = tauw / (0.5 * rho * Uinf^2)
% Re = Uinf * c(=1) / nu
% Cp = (p-pinf) / (0.5 * rho * Uinf^2)
% 
% Mach = 0.2
% Uinf = Mach * a
% Q = 0.5 * rho * Uinf**2
% pref = 101325
% Rgas = 287.05
% rho = 101325 / T(273)
% 
%
% Use dCp/ds instead of DPDS 

% Calculate some parameters first
Ma = 0.2;
Gamma = 1.4;
p_inf = 101325;
Rgas = 287.025;
T_inf = 273;
rho_inf = p_inf / (Rgas * T_inf);
a_inf = sqrt(Gamma * Rgas * T_inf);
U_inf = Ma * a_inf;
c = 1;
Re = 1600000;
mu = (U_inf * c * rho_inf) / Re;

Savetype = "Laminar";
Plot_check = true;
sprintf("Save type is %s", Savetype)
  
% Constants
DOWN_FRAC = 0.01;  % Lower boundary for valid region
UP_FRAC = 0.2;     % Upper boundary for valid region

% Turbulent  = false;
% Laminar    = false;
% Transition = false;
% All        = false;

% Load the data
load N0030_45degsweep_Re1p6M_M0p2_adiab_L14_L16_BLProp.mat
load N0030_45degsweep_Re1p6M_M0p2_adiab_L14_L16_SurfData.mat

% Get variables for easy access
Cpls = SurfData{1,1}.Cpls;
Cfls = SurfData{1,1}.Cfls;
xls  = SurfData{1,1}.xls;
x    = SurfData{1,1}.x;
c    = max(x) - min(x); % Estimated chord length
z    = SurfData{1,1}.z;

% Re   = CFD{1,1}.Re;
% nu   = 4.14e-5; % Estimated from Reynolds numbers
% U_inf = Re * nu / c;

% U_inf = 1;
% nu    = c * U_inf / Re;

dpds = BLProp{1,1}.DPDS;
x100 = BLProp{1,1}.x;

% Set the upper limit
%idx = 500; % The upper limit of the index of x and z
%Cpls = Cpls(1:idx);
%Cfls = Cfls(1:idx);
%x    = x(1:idx);
%z    = z(1:idx);

% Reorder variables
x = [x(401:500) x(1:400)];
z = [z(401:500) z(1:400)];
Cfls = [Cfls(401:500) Cfls(1:400)];
Cpls = [Cpls(401:500) Cpls(1:400)];


% Wall pressure gradient
dCpdx = gradient(Cpls, x);
dCpdz = gradient(Cpls, z);
% dCpds = sqrt(dCpdx.^2 + dCpdz.^2);
% figure(1)
% scatter(x,z);
% 
% figure(2)
% scatter(x100, dpds);

% Calculate the arc length along the airfoil surface
s = zeros(size(x));
for i = 2:length(x)
    % Calculate incremental arc length using Euclidean distance
    ds = sqrt((x(i) - x(i-1))^2 + (z(i) - z(i-1))^2);
    s(i) = s(i-1) + ds;
end

% Calculate pressure gradient along the surface
dCpds_raw = gradient(Cpls, s);

% Apply smoothing to reduce noise
% Method 1: Moving average filter
window_size = 15;  % Adjust window size as needed
dCpds_smooth1 = movmean(dCpds_raw, window_size);

% Method 2: Savitzky-Golay filter (preserves peaks better)
% This is often better for aerodynamic data
frame_length = 23;  % Must be odd, adjust as needed
poly_order = 2;     % Polynomial order, typically 2-4
dCpds_smooth2 = sgolayfilt(dCpds_raw, poly_order, frame_length);
dpds_all = dCpds_smooth2 * 0.5 * rho_inf * U_inf^2;
%
% Plot pressure gradient - original vs smoothed
% figure(2)
% plot(s, dCpds_raw, 'r:', 'LineWidth', 1);  % Original (dotted)
% hold on
% plot(s, dCpds_smooth1, 'b-', 'LineWidth', 1.5);  % Moving average
% plot(s, dCpds_smooth2, 'g-', 'LineWidth', 1.5);  % Savitzky-Golay
% xlabel('Surface arc length (s)');
% ylabel('dC_p/ds');
% title('Pressure Gradient along surface');
% legend('Raw gradient', 'Moving average', 'Savitzky-Golay');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get input/output pairs


% Initialize arrays to store results
inputs = [];
output = [];
flow_type = {};
unnormalized_inputs = [];

% Get the number of data points to process
n_locations = length(data);

% Process each data point
x_investigate = BLProp{1,1}.x;

for idx = 1:n_locations

   

    % Get data for current location
    data_i = data{idx};
    
    % Extract fields based on the structure
    x_loc = x_investigate(idx);  % x location

    
    

    r = data_i.r;              % Wall-normal distance (y)
    up = data_i.up;            % Tangential velocity (U)
    ue = data_i.ue;            % Edge velocity (Ue)
    rho = data_i.rho;
    nu = (mu ./ rho)';
    del99_idx = data_i.del99_idx;  % Boundary layer edge index
    % delta = r(del99_idx);      % Boundary layer thickness (delta99)
    delta = data_i.del95;
    rhoe  = data_i.rhoe;       % Density at edge (rhoe)

    %%%%%%%%%%%% Optional %%%%%%%%%%

    if Savetype == "Turbulent"
     if x_loc < 0.7
        continue;
     end
    elseif Savetype == "Laminar"
      if x_loc > 0.6 || x_loc < 0.25 % This is to remove some stations having issues in postprocessing 
        continue;
      end
     elseif Savetype == "Transition"
      if x_loc < 0.6 || x_loc > 0.7
        continue;
     end
    end


    if Plot_check
    plot_vel_profile(data, x_investigate, idx)
    uiwait(gcf); % Wait until the figure is closed
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Calculate utau (friction velocity)
    % In the absence of direct Cf data, use del_star/theta relationship
    Cf   = interp1(x, Cfls, x_loc, 'linear', 'extrap');
    utau = sqrt(Cf*rho_inf/2) * U_inf;    % Friction velocity
    
    % Calculate pressure gradient
    dPdx = interp1(x, dpds_all , x_loc, 'linear', 'extrap');
    
    % Calculate pressure gradient velocity scale
    up_scale = sign(dPdx) * (abs(nu(1) * dPdx) / rho(1))^(1/3);
    
    % Skip the first point if it is zero
    if up(1) == 0 || r(1) == 0
        up = up(2:end);
        r = r(2:end);
    end
    
    % Find indices for the valid region (0.005*delta < r < 0.25*delta)
    bot_indices = find(r >= DOWN_FRAC*delta & r <= UP_FRAC*delta);
    
    % Skip if no valid points found
    if isempty(bot_indices)
        continue;
    end
    
    % Find velocities at different y locations using helper function
    U2 = find_k_y_values(r(bot_indices), up, r, 3);
    U3 = find_k_y_values(r(bot_indices), up, r, 5);
    U4 = find_k_y_values(r(bot_indices), up, r, 7);
    
    % Calculate pi parameters (inputs)
    pi_1 = up(bot_indices)' .*r(bot_indices) ./ nu(bot_indices);
    pi_2 = up_scale * r(bot_indices) ./ nu(bot_indices);
    pi_3 = U2 .* r(bot_indices) ./ nu(bot_indices);
    pi_4 = U3 .* r(bot_indices) ./ nu(bot_indices);
    pi_5 = U4 .* r(bot_indices) ./ nu(bot_indices);
    
    % Calculate velocity gradient
    dudy = gradient(up, r);
    dudy_1 = dudy(bot_indices)';
    dudy_2 = find_k_y_values(r(bot_indices), dudy, r, 3);
    dudy_3 = find_k_y_values(r(bot_indices), dudy, r, 5);
    
    pi_6 = dudy_1 .* r(bot_indices).^2 ./ nu(bot_indices);
    pi_7 = dudy_2 .* r(bot_indices).^2 ./ nu(bot_indices);
    pi_8 = dudy_3 .* r(bot_indices).^2 ./ nu(bot_indices);
    
    % Calculate output parameter (pi_out = utau * y / nu)
    pi_out = utau * r ./ nu(1);
    pi_out = pi_out(bot_indices);
    
    % Create and stack inputs matrix
    inputs_temp = [pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8];
    inputs = [inputs; inputs_temp];
    
    % Append output array
    output = [output; pi_out];
    
    % Create unnormalized inputs
    n_points = length(bot_indices);
    unnormalized_inputs_temp = zeros(n_points, 11);
    
    for i = 1:n_points
        idx_i = bot_indices(i);
        unnormalized_inputs_temp(i, :) = [
            r(idx_i), ...                       % 1. y coordinate
            up(idx_i), ...                      % 2. U velocity
            nu(idx_i), ...                             % 3. kinematic viscosity
            utau, ...                           % 4. friction velocity
            up_scale, ...                       % 5. pressure gradient velocity
            U2(i), ...                          % 6. U2 velocity
            U3(i), ...                          % 7. U3 velocity
            U4(i), ...                          % 8. U4 velocity
            dudy_1(i), ...                      % 9. dudy1 velocity gradient
            dudy_2(i), ...                      % 10. dudy2 velocity gradient
            dudy_3(i) ...                       % 11. dudy3 velocity gradient
        ];
    end
    
    unnormalized_inputs = [unnormalized_inputs; unnormalized_inputs_temp];
    
    % Create flow type information
    % In MATLAB, we use cell arrays for mixed data types
    flow_type_temp = cell(n_points, 5);
    for i = 1:n_points
        flow_type_temp{i, 1} = 'naca0025';       % 1. Flow type
        flow_type_temp{i, 2} = nu(1);              % 2. Reynolds number (for normalization)
        flow_type_temp{i, 3} = x_loc;           % 3. x coordinate (unnormalized)
        flow_type_temp{i, 4} = delta;           % 4. delta (boundary layer thickness)
        flow_type_temp{i, 5} = 0;          % 5. Albert parameter
    end
    
    % Combine flow_type data
    if isempty(flow_type)
        flow_type = flow_type_temp;
    else
        flow_type = [flow_type; flow_type_temp];
    end
end

% Plot input features
figure(3)
scatter(inputs(:,1), inputs(:,2))
figure(4)
scatter(inputs(:,1), inputs(:,3))

% Define helper function for finding values at locations k*y
function values = find_k_y_values(y_base, U, y_all, k)
    % Function to find velocity values at locations k*y_base
    values = zeros(size(y_base));
    y_target = k * y_base;
    
    for i = 1:length(y_base)
        values(i) = interp1(y_all, U, y_target(i), 'linear', 'extrap');
    end
end

% Save results to file
savename = sprintf('airfoil_bl_pi_fine_%s_parameters.mat', Savetype)
save(savename, 'inputs', 'output', 'flow_type', 'unnormalized_inputs');

