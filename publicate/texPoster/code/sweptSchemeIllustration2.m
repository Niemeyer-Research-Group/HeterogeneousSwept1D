clear
clc
close all

%Full Triangle with tier type.  Problem with full triangle, folding.
%Solution, cut out flux parts.  Proceed as before.
%Triangle to flat and back.  Maybe use numbers.

tpb = 16;
clr = {'r','g'};
index = 1:tpb;
as = [0 100 700 300];
h1 = figure(1);
h2 = figure(2);
h3 = figure(3);
h4 = figure(4);
mksz = 9;
lw = 2;

for k = 1:tpb/2
    for n = 1:2
        for a = k:(tpb+1)-k
            figure(1)
            hold on
            g = a+(n-1)*tpb-1;
            h = plot(g,k-1,strcat('o',clr{n}),'Markersize', mksz, 'MarkerFaceColor',clr{n},'LineWidth',lw);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
        
            figure(2)
            hold on
            g = a+(n-1)*tpb-1;
            h = plot(g,k-1,strcat('o',clr{n}), 'Markersize', mksz+1,'MarkerFaceColor',clr{n},'LineWidth',lw);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end

figure(1)
xlabel('Spatial point')
ylabel('Sub-timestep')
text(-7,4, 'Timestep', 'fontsize', 10)
plot([-8,2*tpb],[3.5, 3.5],'k')
text(-7,8, 'Timestep', 'fontsize', 10)
plot([-8,2*tpb],[7.5, 7.5],'k')
xlim([-8,2*tpb-.5])
ylim([-1,.75*tpb])
plot([-.5,-.5],[-1,2*tpb],'k','Linewidth',lw)
set(gcf,'Position',as)


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

saveas(h1, 'SecondOrderProblemSetting.pdf');

%Saving Effectively
% ----------------

figure(2)
set(gcf,'Position',as)
xlim([-1, 2*tpb+1])
ylim([-1, .75*tpb])

xlabel('Spatial point')
ylabel('Sub-timestep')
x1 = [2,3,12,13,6:9];
y = [0,0,0,0,4,4,4,4];
hold on
plot(x1,y,'Xk','Markersize',mksz+2,'Linewidth',lw+1)
x2 = x1 + 2*(15-x1) +1; 
plot(x2,y,'Xk','Markersize',mksz+2,'Linewidth',lw+1)

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

saveas(h2, 'SecondOrderProblemSettingExes.pdf');

%Saving Effectively
% ----------------
mksz = 1.25*mksz;
lw = 1.25*lw;

for k = 1:tpb/4
    for n = 1:2
        for a = 2*(k-1):((tpb-1)-2*(k-1))
            figure(3)
            hold on
            g = a+(n-1)*(tpb);
            h = plot(g,k-1,strcat('o',clr{n}), 'Markersize', mksz, 'MarkerFaceColor',clr{n},'LineWidth',lw);
            if a<(2*(k-1)+4) || a>((tpb-1)-2*(k-1)-4)
                set(h,'MarkerEdgeColor','k')
            end
            figure(4)
            hold on
            g = a+(n-1)*(tpb);
            if a<(2*(k-1)+4) || a>((tpb-1)-2*(k-1)-4)
                h = plot(g,k-1,strcat('o',clr{n}), 'Markersize', mksz, 'MarkerFaceColor',clr{n},'LineWidth',lw);
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end


base = tpb + 2;
ht = base/2;
ht2 = tpb/2;
circ = [4:11, 20:27];
hold on
c = 1;
y = 1:4;
for k = ht2:-2:1
    for n = 1:2
        for a = k-1:(base-k)
            g = mod((a+(n-1)* tpb-1)-ht2,2*tpb);
            h = plot(g,y(c),strcat('o',clr{n}), 'Markersize', mksz, 'MarkerFaceColor',clr{n},'LineWidth',lw);
            if k == 2 && sum(g == circ)> 0
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
    c = c+1;
end

figure(3)
as = as/1.25;
set(gcf,'Position',as)
xlim([-1,2*tpb-.5])
ylim([-1,.5*tpb])
xlabel('Spatial point')
ylabel('Sub-timestep')

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

saveas(h3, 'SecondOrderStepOne.pdf');

%Saving Effectively
% ----------------

figure(4)
as = as+50;
set(gcf,'Position',as)
xlim([-1,2*tpb-.5])
ylim([-1,.5*tpb])


xlabel('Spatial point')
ylabel('Sub-timestep')
y = 5:8;
c = 1;
for k = 2:tpb/4
    for n = 1:2
        for a = 2*(k-1):((tpb-1)-2*(k-1))
            g = mod((a+(n-1)* tpb-1)-ht2+1,2*tpb);
            h = plot(g,y(c),strcat('o',clr{n}), 'Markersize', mksz, 'MarkerFaceColor',clr{n},'LineWidth',lw);
            if a<(2*(k-1)+4) || a>((tpb-1)-2*(k-1)-4)
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
    c = c+1;
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

saveas(h4, 'SecondOrderStepTwo.pdf');

%Saving Effectively
% ----------------
