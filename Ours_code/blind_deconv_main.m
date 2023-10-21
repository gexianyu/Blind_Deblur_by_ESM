function [k, w1, w2, alpha, S] = blind_deconv_main(blur_B, k, ...
                                    w1, w2, alpha, threshold, opts)
% derivative filters
dx = [-1 1; 0 0];
dy = [-1 0; 1 0];
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2013-08-11
H = size(blur_B,1);    W = size(blur_B,2);
blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size([H W]+size(k)-1));
blur_B_tmp = blur_B_w(1:H,1:W,:);
Bx = conv2(blur_B_tmp, dx, 'valid');
By = conv2(blur_B_tmp, dy, 'valid');
Bxx = conv2(conv2(blur_B_tmp,dx,'valid'),dx,'valid');
Byy = conv2(conv2(blur_B_tmp,dy,'valid'),dy,'valid');
Bxy = conv2(conv2(blur_B_tmp,dx,'valid'),dy,'valid');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:opts.xk_iter
   S = L0Restoration_ISR(blur_B, k, w1, w2, alpha, opts.p_norm);
   [latent_x, latent_y, threshold]= threshold_pxpy_v1(S, max(size(k)), threshold);
   [latent_xx, latent_xy,~] = threshold_pxpy_v1(latent_x, max(size(k)));
   [~,latent_yy,~] = threshold_pxpy_v1(latent_y, max(size(k)));
   k = estimate_psf_ISR(Bx, By, latent_x, latent_y, Bxx, Byy, Bxy,...
      latent_xx, latent_yy, latent_xy, 2, w1, w2, k);
  fprintf('pruning isolated noise in kernel...\n');
  CC = bwconncomp(k,8);
  for ii=1:CC.NumObjects
      currsum=sum(k(CC.PixelIdxList{ii}));
      if currsum<.1 
          k(CC.PixelIdxList{ii}) = 0;
      end
  end
  k(k<0) = 0;
  k=k/sum(k(:));
  if w1~=0
      w1 = max(w1/1.1, 1e-4);
  else
      w1 = 0;
  end
  if w2~=0
      w2 = max(w2/1.1, 1e-6);
  else
      w2 = 0;
  end
  if alpha~=0
      alpha = max(alpha/1.1, 1e-4);
  else
      alpha = 0;
  end
  S(S<0) = 0;
  S(S>1) = 1;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  figure(1); 
  subplot(1,3,1); imshow(blur_B,[]); title('Blurred image');
  subplot(1,3,2); imshow(S,[]);title('Interim latent image');
  subplot(1,3,3); imshow(k,[]);title('Estimated kernel');
  drawnow;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
k(k<0) = 0;  
k = k ./ sum(k(:));
