clear
clc
close all

fc = csvread('tEulerC.csv',1,0);
fs = csvread('tEulerS.csv',1,0);

interpc = scatteredInterpolant(fc(:,1:end-1), fc(:,end));
interps = scatteredInterpolant(fs(:,1:end-1), fs(:,end));
interps.ExtrapolationMethod = 'nearest';
interpc.ExtrapolationMethod = 'nearest';
x = unique(fc(:, 1));
y = unique(fc(:, 2)); 
z = unique(fc(:, 3));

xa = linspace(x(2), x(end-1), 100);
ya = linspace(y(2), y(end-1), 100);
za = linspace(z(2), z(end-1), 100);

nxyz = zeros(1e6, 3);
cnt = 1;

for xx = 1:length(xa)
    for yy = 1:length(ya)
        for zz = 1:length(za)
            nxyz(cnt,:) = [xa(xx), ya(yy), za(zz)];
            cnt = cnt+1;
        end
    end
end
nc = interpc(nxyz);
ns = interps(nxyz);

