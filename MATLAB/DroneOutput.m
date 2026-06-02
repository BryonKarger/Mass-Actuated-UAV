out = sim("DroneSim");

state_ts = out.stateLog;      % if using Single Simulation Output

t = state_ts.Time;
X = state_ts.Data;

% Remove singleton dimensions if Simulink gives you 3D data
X = squeeze(X);

% Make sure X is [samples x states]
if size(X,2) ~= 7 && size(X,1) == 7
    X = X.';
end

X;

x     = X(:,1);
y     = X(:,3);
theta = X(:,5);
d     = X(:,7);
T     = X(:,8);

% Create table
T = table(t, x, y, theta, d, T, ...
    'VariableNames', {'time','x','y','theta','d', 'thrust'});

% Write to CSV
writetable(T, "drone_log.csv");