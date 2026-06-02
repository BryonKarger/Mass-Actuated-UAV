out = sim("DroneSim");

state_ts = out.stateLog;      % if using Single Simulation Output

t = state_ts.Time;
X = state_ts.Data;

if size(X,2) ~= 7 && size(X,1) == 7
    X = X.';
end

x     = X(:,1);
y     = X(:,3);
theta = X(:,5);
d     = X(:,7);

figure;
axis equal;
grid on;
hold on;

xlabel("x position, m");
ylabel("y position, m");
title("Static Thrust Vector Drone Animation");

xlim([min(x)-1, max(x)+1]);
ylim([min(y)-1, max(y)+1]);

bodyLength = 0.4;
bodyWidth  = 0.25;

bodyShape = [
    0,           bodyLength;
    -bodyWidth,  -bodyLength;
    bodyWidth,  -bodyLength
    ]';

dronePatch = patch(NaN, NaN, "k");
pathLine = plot(NaN, NaN, "--");

comMarker = plot(NaN, NaN, "o", "MarkerFaceColor", "k");

for k = 1:5:length(t)

    R = [
        cos(theta(k)), -sin(theta(k));
        sin(theta(k)),  cos(theta(k))
        ];

    bodyWorld = R*bodyShape + [x(k); y(k)];

    set(dronePatch, ...
        "XData", bodyWorld(1,:), ...
        "YData", bodyWorld(2,:));

    set(pathLine, ...
        "XData", x(1:k), ...
        "YData", y(1:k));

    comOffsetBody = [d(k); 0];
    comWorld = R*comOffsetBody + [x(k); y(k)];

    set(comMarker, ...
        "XData", comWorld(1), ...
        "YData", comWorld(2));

    drawnow limitrate;
end