%% fig_1_2.m
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));

%% Get IdVg data for nFET dev1_100_100(2) ; 3_30_2025 3_40_14 PM

csvFile = fullfile(scriptDir, 'EE312_Si_nFET_compiled.csv');

fid = fopen(csvFile, 'r');
headerLine = fgetl(fid);
fclose(fid);
headers = strsplit(headerLine, ',');

vgsCol = find(strcmp(headers, ...
    'nFET_IdVg_Vdpar [dev1_100_100(2) ; 3_30_2025 3_40_14 PM].csv_VGS'));
idCol = find(strcmp(headers, ...
    'nFET_IdVg_Vdpar [dev1_100_100(2) ; 3_30_2025 3_40_14 PM].csv_ID'));

data = readmatrix(csvFile);

VGS_Si_nFET_full = data(:, vgsCol);
ID_Si_nFET_full = data(:, idCol);

%% Get IdVg data for IdVg_main_20260618_145926_DieR_3_DieC_0_L_W_25u_100u_ID_C

csvFile2 = fullfile(scriptDir, 'ITO_t2nm_L25um_W100um_compiled.csv');

fid = fopen(csvFile2, 'r');
headerLine2 = fgetl(fid);
fclose(fid);
headers2 = strsplit(headerLine2, ',');

vgCol = find(strcmp(headers2, '0_3_3_Vg'));
idCol2 = find(strcmp(headers2, '0_3_3_Id_0.05_2'));

data2 = readmatrix(csvFile2);

Vg_OS_nFET_full = data2(:, vgCol);
Id_OS_nFET_full = data2(:, idCol2);

%% Figure 1: Plot IdVg for OS nFET for VTR illustration

% Vg sweep is bidirectional (forward then reverse); keep only the forward sweep
[~, pkIdx] = max(Vg_OS_nFET_full);
Vg_OS_nFET = Vg_OS_nFET_full(1:pkIdx);
Id_OS_nFET = Id_OS_nFET_full(1:pkIdx);

W = 100e-6; L = 25e-6; Vds = 0.05;

fig1 = figure('Name','Id_vs_Vg_OS_nFET', 'Visible', 'off');

yyaxis left; % Left y-axis
semilogy(Vg_OS_nFET, abs(Id_OS_nFET)/W, 'LineWidth', 3.5, 'Color', [0 0 1]);
ylabel('I_{D} [log scale]', 'FontSize', 18, 'FontWeight', 'bold');
yticks([])

yyaxis right; % Right y-axis
plot(Vg_OS_nFET, Id_OS_nFET/W, 'LineWidth', 3.5, 'Color', [0 0 1]);
ylabel('I_{D} [linear scale]', 'FontSize', 18, 'FontWeight', 'bold');
yticks([])

xlabel('V_{GS}', 'FontSize', 18, 'FontWeight', 'bold');

ax = gca;
ax.FontSize = 18;
ax.FontWeight = "bold";
ax.LineWidth = 3.5;
ax.YAxis(1).Color = [0 0 0];
ax.YAxis(2).Color = [0 0 0];
xlim([min(Vg_OS_nFET) max(Vg_OS_nFET)])
xticks([])

% Use tight layout
set(gca, 'LooseInset', max(get(gca,'TightInset'), 0.02))

% Set the figure and axes background to transparent
set(gcf, 'Color', 'none');
set(gca, 'Color', 'none');

% Save the figure as SVG (vector, transparent background) in the script directory
exportgraphics(fig1, fullfile(scriptDir, 'fig1.svg'), 'BackgroundColor', 'none');

