clear
clc
close all

fC = csvread('tHeatCc.csv');
fS = csvread('tHeatSc.csv');
fC = sortrows(fC, 2);
fS = sortrows(fS, 2);
XS = fS(:,1:end-1);
XC = fC(:,1:end-1);
yS = fS(:,end);
yC = fC(:,end);
typ = 'interactions'; %'quadratic'

glmS = fitlm(XS, yS, typ)
glmC = fitlm(XC, yC, typ)

% utpb = unique(fS(:,1));
% ugpuA = unique(fS(:,3));
% ugpuA = 1:5
% Leg={};
% for n = 1:length(ugpuA)
%     Leg{n} = num2str(ugpuA(n));
% end
% rng = [min(fS(:,2)), max(fS(:,2))];
% figure
% for k = 1:length(utpb)
%     subplot(2, 2, k)
%     hold on
%     for n = 1:length(ugpuA)
%         cnd = fS(:, 1) == utpb(k) & fS(:,3) == ugpuA(n);
%         x = fS(cnd, 2);
%         y = fS(cnd, 4);
%         semilogx(x, y, 'LineWidth', 2);
%     end
%     title(sprintf('%.d threads per block', utpb(k)));
%     xlabel('Grid size')
%     ylabel('time per timestep (\mu s)')
%     set(gca, 'XScale', 'log')
%     xlim(rng)
%     ylim([2, 300])
%     grid on
% end
% 
% h = legend(Leg);
% v = get(h,'title');
% set(v,'string','GPU affinity');
% 
% figure
% for k = 1:length(utpb)
%     subplot(2, 2, k)
%     hold on
%     for n = 1:length(ugpuA)
%         cnd = fC(:, 1) == utpb(k) & fC(:,3) == ugpuA(n);
%         x = fC(cnd, 2);
%         y = fC(cnd, 4);
%         semilogx(x, y, 'LineWidth', 2);
%     end
%     title(sprintf('%.d threads per block', utpb(k)));
%     xlabel('Grid size')
%     ylabel('time per timestep (\mu s)')
%     set(gca, 'XScale', 'log')
%     ylim([2, 300])
%     xlim(rng)
%     grid on
% end
% 
% h = legend(Leg);
% v = get(h,'title');
% set(v,'string','GPU affinity');
% 
% % figure
% % subplot(2, 1, 1)
% % hold on
% % semilogx(ugd, sDbst, 'LineWidth', 2);
% % semilogx(ugd, cDbst, 'LineWidth', 2);
% % xlabel('Grid size')
% % ylabel('time per timestep (\mu s)')
% % set(gca, 'XScale', 'log')
% % title('Best Runs')
% % legend('Swept', 'Classic')
% % xlim([gdm ,mx])
% % grid on
% % 
% % subplot(2, 1, 2)
% % semilogx(ugd, dspdUp, 'LineWidth', 2);
% % xlabel('Grid size')
% % ylabel('Speedup')
% % set(gca, 'XScale', 'log')
% % xlim([gdm ,mx])
% % 
% % grid on



