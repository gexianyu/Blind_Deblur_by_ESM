clc;
clear;
close all;
dirname_blurry = 'Blurry_images';
dirname_results = 'Results';
if ~exist(dirname_results,'dir')
    mkdir(dirname_results);
end
addpath(genpath('./Ours_code'));
addpath(genpath('./Outer_code'));
opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %%the iterations
opts.k_thresh = 20; %%threshold
opts.gamma_correct = 1.0; %%gamma correction


%% runtime test
fig = 'Blurry2_4_200'; opts.kernel_size = 21; opts.p_norm = 0.6;    
% fig = 'Blurry2_4_400'; opts.kernel_size = 21; opts.p_norm = 0.6;
% fig = 'Blurry2_4_800'; opts.kernel_size = 21; opts.p_norm = 0.6;

%% Inputs
% fig = 'im01_ker01_levin'; opts.kernel_size = 21; opts.p_norm = 0.6;
% fig = 'im01_ker01_blur'; opts.kernel_size = 21; opts.p_norm = 0.6;
% fig = 'im04_ker01_blur'; opts.kernel_size = 21; opts.p_norm = 0.4;
% fig = 'boat2'; opts.kernel_size = 51; opts.p_norm = 0.3;
% fig = 'postcard'; opts.kernel_size = 101; opts.p_norm = 0.5;
% fig = 'blur_3_ker_4'; opts.kernel_size = 101; opts.p_norm = 0.6;
% fig = '35_4_blurred'; opts.kernel_size = 51; opts.p_norm = 0.6;
% fig = '39_6_blurred'; opts.kernel_size = 51; opts.p_norm = 0.9;
% fig = '44_1_blurred'; opts.kernel_size = 51; opts.p_norm = 0.2;
% fig = 'people_02_kernel_03'; opts.kernel_size = 75; opts.p_norm = 0.3;
% fig = 'saturated_01_kernel_04'; opts.kernel_size = 101; opts.p_norm = 0.8;


%% Prepare the blurry image
filename = sprintf('%s/%s.png',dirname_blurry,fig);
if ~exist(filename,'file')
    filename = sprintf('%s/%s.jpg',dirname_blurry,fig);
end
y = imread(filename);
if size(y,3)==3
    yg = im2double(rgb2gray(y));
else
    yg = im2double(y);
end
%% Estimatation of kernel
tic;
[kernel, interim_latent] = blind_deconv(yg, 4e-3, 4e-4, 4e-3, opts);
toc
%% Non-blind deconvolution
y = im2double(y);
Latent = ringing_artifacts_removal(y, kernel, 0.001, 2e-4, 1);
%% Save the results
k = kernel - min(kernel(:)); k = k./max(k(:));
Latent_name = [fig,'_Latent'];
k_name = [fig,'_kernel'];
imwrite(Latent, sprintf('%s/%s.png',dirname_results,Latent_name));
imwrite(k, sprintf('%s/%s.png',dirname_results,k_name));
%% 
rmpath(genpath('./Ours_code'));
rmpath(genpath('./Outer_code'));

