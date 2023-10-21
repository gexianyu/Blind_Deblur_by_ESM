function S = L0Restoration_ISR(Im, kernel,w1,w2,alpha,p_norm)

if ~exist('kappa','var')
    kappa = 2.0;
end
%% pad image
H = size(Im,1);    W = size(Im,2);
Im = wrap_boundary_liu(Im, opt_fft_size([H W]+size(kernel)-1));
%%
S = Im;
betamax = 1e5;
fx = [1 -1]; fy = [1; -1]; 
fxx = [1 -2 1]; fyy = [1;-2;1]; fxy = [1 -1; -1 1];
[N,M,D] = size(Im); sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
otfFxx = psf2otf(fxx,sizeI2D);
otfFyy = psf2otf(fyy,sizeI2D);
otfFxy = psf2otf(fxy,sizeI2D);
%%
KER = psf2otf(kernel,sizeI2D);
Den_KER = abs(KER).^2;
%%
Denormin3 = abs(otfFx).^2 + abs(otfFy ).^2;
if D>1
    Denormin3 = repmat(Denormin3,[1,1,D]);
    KER = repmat(KER,[1,1,D]);
    Den_KER = repmat(Den_KER,[1,1,D]);
end
Normin1 = conj(KER).*fft2(S);
%% 
delta1 = 2 * w1; delta2 = 2 * w2; delta3 = 2 * alpha;
B_x = [diff(Im,1,2), Im(:,1,:) - Im(:,end,:)];
B_y = [diff(Im,1,1); Im(1,:,:) - Im(end,:,:)];
Bf = fft2(Im);
B_xx = real(ifft2(Bf.*otfFxx)); 
B_xy = real(ifft2(Bf.*otfFxy)); 
B_yy = real(ifft2(Bf.*otfFyy)); 

KG_x = otfFx.*KER; KG_y = otfFy.*KER;
KG1 = abs(KG_x).^2 + abs(KG_y).^2;
KG_xx = otfFxx.*KER; KG_yy = otfFyy.*KER; KG_xy = otfFxy .*KER;
KG2 = abs(KG_xx).^2 + abs(KG_yy).^2 + abs(KG_xy).^2;

while delta3 < betamax
    
    Denormin = Den_KER + delta3 * Denormin3 + delta1 * KG1 + delta2 * KG2;
    S_x = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
    S_y = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
    Sf = fft2(S);
    S_xx = real(ifft2(Sf.*otfFxx));
    S_xy = real(ifft2(Sf.*otfFxy));
    S_yy = real(ifft2(Sf.*otfFyy));
    
    %%estimate f1
    if delta1 ~= 0
        f1_x = B_x - real(ifft2(fft2(S_x).*KER));
        f1_y = B_y - real(ifft2(fft2(S_y).*KER));
        
        f1_x = sign(f1_x).*max(abs(f1_x) - w1/(2*delta1),0); 
        x_index = f1_x.^2 < w1/delta1; f1_x(x_index) = 0;
        f1_y = sign(f1_y).*max(abs(f1_y) - w1/(2*delta1),0);
        y_index = f1_y.^2 < w1/delta1; f1_y(y_index) = 0;
        
        Normin3 = conj(KG_x).*fft2(B_x - f1_x) + ...
            conj(KG_y).*fft2(B_y - f1_y);
    else
        Normin3 = zeros(size(B_x));
    end
    
    %%estimate f2
    if delta2 ~= 0
        f2_xx = B_xx - real(ifft2(fft2(S_xx).*KER)); 
        f2_xy = B_xy - real(ifft2(fft2(S_xy).*KER));
        f2_yy = B_yy - real(ifft2(fft2(S_yy).*KER));
        
        f2_xx = sign(f2_xx).*max(abs(f2_xx) - w2/(2*delta2),0); 
        xx_index = f2_xx.^2 < w2/delta2; f2_xx(xx_index) = 0;
        f2_xy = sign(f2_xy).*max(abs(f2_xy) - w2/(2*delta2),0); 
        xy_index = f2_xy.^2 < w2/delta2; f2_xy(xy_index) = 0;
        f2_yy = sign(f2_yy).*max(abs(f2_yy) - w2/(2*delta2),0); 
        yy_index = f2_yy.^2 < w2/delta2; f2_yy(yy_index) = 0;
          
        Normin4 = conj(KG_xx).*fft2(B_xx - f2_xx) + ...
            conj(KG_xy).*fft2(B_xy - f2_xy) + ...
            conj(KG_yy).*fft2(B_yy - f2_yy);
    else
        Normin4 = zeros(size(B_xx));
    end
  
    %%%estimate g
    [g_x, ~] = solve_L0p(S_x, alpha/delta3, p_norm);
    [g_y, ~] = solve_L0p(S_y, alpha/delta3, p_norm);
    Normin2 = [g_x(:,end,:) - g_x(:, 1,:), -diff(g_x,1,2)]...
              + [g_y(end,:,:) - g_y(1, :,:); -diff(g_y,1,1)];
    
    %%%%estimate u
    FS = (Normin1 + delta3*fft2(Normin2) + delta1*Normin3 + delta2*Normin4)./Denormin;
    S = real(ifft2(FS));
    delta1 = delta1 * kappa;
    delta2 = delta2 * kappa;
    delta3 = delta3 * kappa;
end
S = S(1:H, 1:W, :);
end
