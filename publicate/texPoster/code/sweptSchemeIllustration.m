clear
clc
close all

tpb = 16;
clr = {'r', 'g'};
index = 1:tpb;
h1 = figure(1);
h2 = figure(2);
mksz = 9;

for k = 1:tpb/2
    for n = 1:2
        for a = k:(tpb+1)-k
            
            figure(1)
            hold on
            g = a+(n-1)*tpb-1;
            %All points
            h = plot(g,k-1,strcat('o',clr{n}),'Markersize', mksz, 'MarkerFaceColor',clr{n}, 'LineWidth', 2);
            %Edge Points
            if a<(k+2) || a>(tpb-(k+1))
                set(h, 'MarkerEdgeColor','k')
            end

            figure(2)
            hold on       
            if a<(k+2) || a>(tpb-(k+1))
                h = plot(g,k-1,strcat('o',clr{n}), 'Markersize', mksz, 'MarkerFaceColor', clr{n},'LineWidth',2);
                set(h, 'MarkerEdgeColor','k')
                
            end
        end
    end
end

mw = [0 100 700 300];
set(gcf,'Position', mw)
xlabel('Spatial point')
ylabel('Sub-timestep')
% xt = get(gcf, 'XTickLabel');
% disp(xt)

figure(1)
set(gcf, 'Position', mw)
annotation('textbox', [0.26, 0.45, 0.3, 0.3],'String', 'Node 0', 'FitBoxToText','on');
annotation('textbox', [0.71, 0.45, 0.3, 0.3],'String', 'Node 1', 'FitBoxToText','on');
plot([15.5, 15.5], [-1, tpb],'k')
xlabel('Spatial point')
ylabel('Sub-timestep')
xlim([-1,2*tpb-.5])
ylim([-1,tpb])

%Saving Effectively
% ----------------
ti = get(gca,'TightInset');
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);

set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

set(gcf, 'PaperUnits','centimeters');
set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);

saveas(h1, 'FirstOrderStepOne.pdf');

%Saving Effectively
% ----------------

figure(2)
xlim([-1,2*tpb-.5])
ylim([-1,tpb])
xlabel('Spatial point')
ylabel('Sub-timestep')

hold on
x = [.20,.12; .41,.49; .65,.57; .85, .93];
y = [.25,.35];


base = tpb + 2;
ht = base/2;
ht2 = tpb/2;

circ = [6:9,22:25];

for k = ht2:-1:1
    for n = 1:2
        for a = k:(base-k-1)
            g = mod((a+(n-1)* tpb-1)-ht2,2*tpb);
            h = plot(g,ht-k,strcat('o',clr{n}), 'Markersize', mksz, 'MarkerFaceColor',clr{n},'LineWidth',2);
            if k == 1 && sum(g == circ)> 0
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end
annotation('textarrow',x(1,:),y,'String','L_{0} -> R_{0}','LineWidth',1.5,'FontSize',12);
annotation('textarrow',x(2,:),y,'String','R_{0} -> L_{1}','LineWidth',1.5,'FontSize',12);
annotation('textarrow',x(3,:),y,'String','L_{1} -> R_{1}','LineWidth',1.5,'FontSize',12);
annotation('textarrow',x(4,:),y,'String','R_{1} -> L_{0}','LineWidth',1.5,'FontSize',12);

for k = 2:tpb/2
    for n = 1:2
        for a = k:(tpb+1)-k
            g = mod((a+(n-1)* tpb-1)-ht2,2*tpb);
            h = plot(g,ht2+(k-1),strcat('o',clr{n}), 'Markersize', mksz,'MarkerFaceColor',clr{n},'LineWidth',2);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end


%Saving Effectively
% ----------------
ti = get(gca,'TightInset');
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);

set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

set(gcf, 'PaperUnits','centimeters');
set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);

saveas(h2,'FirstOrderStepTwo.pdf');

%% Last part
h3 = figure(3);
figure(3);
for k = 1:4
    for n=1:32
        
        hold on
        an = (n>16) + 1;
        h = plot(n-1,k-1,strcat('o',clr{an}),'Markersize', mksz, 'MarkerFaceColor',clr{an}, 'LineWidth', 3);
        if (n<19 && n>14)
            set(h, 'MarkerEdgeColor','k')
        end
    end
end
%% Three

mw = [0 100 700 300];
annotation('line',[0.51, 0.54], [0.60, 0.78], 'LineWidth', 5)
annotation('arrow',[0.54, 0.57], [0.78, 0.60], 'LineWidth', 5)
annotation('line',[0.54, 0.51], [0.60, 0.78], 'LineWidth', 5)
annotation('arrow',[0.51, 0.48], [0.78, 0.60], 'LineWidth', 5)
set(gcf,'Position',mw)
xlabel('Spatial point')
ylabel('Sub-timestep')
xlim([-0.5, 31.5])
ylim([-0.5, 6.5])
ti = get(gca,'TightInset');
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);

set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

set(gcf, 'PaperUnits','centimeters');
set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);

saveas(h3,'ClassicScheme.pdf');
