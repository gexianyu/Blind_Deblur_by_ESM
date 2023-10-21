blurred = imread('./images/26.png');
blurred = double(blurred)/255;

psf = double(imread('./results/26_psf.png'));
psf = psf./sum(psf(:));

reg_strength = 0.004;  % use large number for more smooth result

deblurred = deconv_RL_sat(blurred,psf,reg_strength);
figure;imshow([blurred,deblurred]);