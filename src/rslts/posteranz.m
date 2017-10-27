clear
clc
close all

fC = csvread('tHeatCc.csv');
fS = csvread('tHeatSc.csv');

ptS = [fS(:,1), fS(:,3), fS(:,2)];
ptC = [fC(:,1), fC(:,3), fC(:,2)];

siS = scatteredInterpolant(ptS, fS(:, end), 'natural', 'nearest');
siC = scatteredInterpolant(ptC, fC(:, end), 'natural', 'nearest');

gdi = [];
utpb = unique(fS(:,1));
for k = 1:length(utpb)
    gs = fS(fS(:,1) == utpb(k), 2);
    gc = fC(fC(:,1) == utpb(k), 2);
    gdi = [gdi, min(gs), min(gc)];
end

gdm = max(gdi);
ugd = unique([fS(:,2); fC(:,2)]);
ugd = ugd(ugd>gdm);
ugpuA = unique(fS(:,3));
% ugpuA = 1:5;
ugd = ugd(ugd<=2^19);
[G, A, T] = meshgrid(ugd, ugpuA, utpb);

Sest = siS(T, A, G);
Cest = siC(T, A, G);

[sBst, sL] = min(Sest, [], 3);
[cBst, cL] = min(Cest, [], 3);

sDbst = min(sBst, [], 1);
cDbst = min(cBst, [], 1);

spdUp = cBst./sBst;
dspdUp = cDbst./sDbst;
Leg={};
for n = 1:length(ugpuA)
    Leg{n} = num2str(ugpuA(n));
end
mx = max(G(1,:,1));
figure
for k = 1:length(utpb)
    subplot(2, 2, k)
    hold on
    for n = 1:length(ugpuA)
        semilogx(G(n, :, k), Sest(n, :, k), 'LineWidth', 2);
    end
    title(sprintf('%.d threads per block', utpb(k)));
    xlabel('Grid size')
    ylabel('time per timestep (\mu s)')
    set(gca, 'XScale', 'log')
    xlim([gdm ,mx])
    ylim([2, 300])
    grid on
end

h = legend(Leg);
v = get(h,'title');
set(v,'string','GPU affinity');

figure
for k = 1:length(utpb)
    subplot(2, 2, k)
    hold on
    for n = 1:length(ugpuA)
        g = G(n, :, k);
        c = Cest(n, :, k);
        semilogx(g, c, 'LineWidth', 2);
    end
    title(sprintf('%.d threads per block', utpb(k)));
    xlabel('Grid size')
    ylabel('time per timestep (\mu s)')
    set(gca, 'XScale', 'log')
    xlim([gdm ,mx])
    ylim([2, 300])
    grid on
end

h = legend(Leg);
v = get(h,'title');
set(v,'string','GPU affinity');

wn = 6;
figure
hold on
semilogx(ugd, sDbst, 'LineWidth', wn);
semilogx(ugd, cDbst, 'LineWidth', wn);
xlabel('Grid size', 'FontSize', 16)
ylabel('time per timestep (\mus)', 'FontSize', 16)
set(gca, 'XScale', 'log')
set(gca, 'FontSize', 16)
legend({'Swept', 'Classic'}, 'Location', 'NorthWest', 'FontSize', 16)
xlim([gdm ,mx])
ti = get(gca,'TightInset');
adj = 0.025;
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1)-adj 1-ti(4)-ti(2)-adj]);
grid on

figure
semilogx(ugd, dspdUp, 'LineWidth', wn);
xlabel('Grid size', 'FontSize', 16)
ylabel('Speedup', 'FontSize', 16)
set(gca, 'XScale', 'log')
set(gca, 'FontSize', 16)
xlim([gdm ,mx])
ti = get(gca,'TightInset');
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1)-adj 1-ti(4)-ti(2)-adj]);
grid on

