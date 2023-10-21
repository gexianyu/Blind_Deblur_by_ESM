function psf = estimate_psf_ISR(Bx, By, latent_x, latent_y,Bxx,Byy,Bxy,latent_xx,latent_yy,latent_xy,weight,w1, w2, k_prev)

    latent_xf = fft2(latent_x); latent_yf = fft2(latent_y);
    Bx_f = fft2(Bx); By_f = fft2(By);
    latent_xxf = fft2(latent_xx);
    latent_yyf = fft2(latent_yy);
    latent_xyf = fft2(latent_xy);
    psf_size = size(k_prev);
    phi1 = 2 * w1; phi2 = 2 * w2;
    psf = k_prev;
    iter_max = 5;
    %% 5 is not the necessary iteration step, 
    %% you can replace it with smaller figure for faster processing time.
    for iter = 1:iter_max
        %%estimate h1
        if phi1 ~= 0
            psf_f1 = psf2otf(psf,size(latent_x));
            h1_x = Bx - real(ifft2(latent_xf.*psf_f1));
            h1_y = By - real(ifft2(latent_yf.*psf_f1));
            
            h1_x = sign(h1_x).*max(abs(h1_x) - w1/(2*phi1),0); 
            x_index = h1_x.^2 < w1/phi1; h1_x(x_index) = 0;
            h1_y = sign(h1_y).*max(abs(h1_y) - w1/(2*phi1),0);
            y_index = h1_y.^2 < w1/phi1; h1_y(y_index) = 0;

            temp1 = conj(latent_xf).*fft2(Bx - h1_x) ...
                 + conj(latent_yf).*fft2(By - h1_y);
        else
            temp1 = zeros(size(latent_x));
        end
        
        %%estimate h2
        if phi2 ~= 0
            psf_f2 = psf2otf(psf,size(latent_xx));
            h2_xx = Bxx - real(ifft2(latent_xxf.*psf_f2));
            h2_yy = Byy - real(ifft2(latent_yyf.*psf_f2));
            h2_xy = Bxy - real(ifft2(latent_xyf.*psf_f2));
            
            h2_xx = sign(h2_xx).*max(abs(h2_xx) - w2/(2*phi2),0); 
            xx_index = h2_xx.^2 < w2/phi2; h2_xx(xx_index) = 0;
            h2_yy = sign(h2_yy).*max(abs(h2_yy) - w2/(2*phi2),0); 
            yy_index = h2_yy.^2 < w2/phi2; h2_yy(yy_index) = 0;
            h2_xy = sign(h2_xy).*max(abs(h2_xy) - w2/(2*phi2),0); 
            xy_index = h2_xy.^2 < w2/phi2; h2_xy(xy_index) = 0;
            
             temp2 = conj(latent_xxf).*fft2(Bxx-h2_xx) ...
                 + conj(latent_yyf).*fft2(Byy-h2_yy) ...
                 + conj(latent_xyf).*fft2(Bxy-h2_xy);
        else
            temp2 = zeros(size(latent_xx));
        end
        
        b_f1 = phi1 * temp1 + conj(latent_xf).*Bx_f + conj(latent_yf).* By_f;
        b1 = real(otf2psf(b_f1, psf_size));
        b_f2 = phi2 * temp2;
        b2 = real(otf2psf(b_f2, psf_size));
        b = b1 + b2;
        p.m1 = (1 + phi1) * (conj(latent_xf) .* latent_xf + conj(latent_yf) .* latent_yf);
        p.m2 = phi2 * (conj(latent_xxf).*latent_xxf + conj(latent_yyf).*latent_yyf + conj(latent_xyf).*latent_xyf);  
        p.img_size1 = size(Bx);
        p.img_size2 = size(Bxx);
        p.psf_size = psf_size;
        p.lambda = weight;
        psf = conjgrad(psf, b, 8, 1e-5, @compute_Ax, p);
        phi1 = phi1 * 2;
        phi2 = phi2 * 2;
    end
    psf(psf < max(psf(:))*0.05) = 0;
    psf = psf / sum(psf(:));
end

function y = compute_Ax(x, p)
    x_f1 = psf2otf(x, p.img_size1);
    y1 = otf2psf(p.m1 .* x_f1, p.psf_size);
    x_f2 = psf2otf(x, p.img_size2);
    y2 = otf2psf(p.m2 .* x_f2, p.psf_size);
    y = y1 + y2 + p.lambda * x;
end

