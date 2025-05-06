
function plot_vel_profile(data, x_investigate, idx)

% idx = 5 cluster issue
% other indices many few data inside; seems to be needing much finer grid
% closer to the wall

% idx = 1;
    data_i = data{idx};
    
    % Extract fields based on the structure
    x_loc = x_investigate(idx);  % x location
    r = data_i.r;              % Wall-normal distance (y)
    up = data_i.up;            % Tangential velocity (U)
    ue = data_i.ue;            % Edge velocity (Ue)
    del99_idx = data_i.del99_idx;  % Boundary layer edge index
    % delta = r(del99_idx);      % Boundary layer thickness (delta99)
    delta = data_i.del95;
    rhoe  = data_i.rhoe;       % Density at edge (rhoe)

figure(1)
plot(r, up, '-o');
xlabel("Wall normal distance")
ylabel("Wall parallel velocity")
xline(delta);
title(sprintf('x=%f, idx=%d', x_loc, idx))

figure(2)
plot(r, up, '-o');
xlim([0 delta])
xlabel("Wall normal distance")
ylabel("Wall parallel velocity")
xline(delta);
title(sprintf('x=%f Zoomed in, idx=%d', x_loc, idx))

end