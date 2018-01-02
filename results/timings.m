clear
clc
close all

notefile = fileread('notes.json');
jj = jsondecode(notefile);
fname = fieldnames(jj);
for k = 1:length(fname)
    f = fname{k}
end

h5disp('rawResults.h5', strcat('/',fname{end}))
