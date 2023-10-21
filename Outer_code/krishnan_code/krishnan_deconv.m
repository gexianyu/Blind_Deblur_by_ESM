function x = krishnan_deconv(y,kernel,lambda, alpha)
    
    kernel_pad = floor((size(kernel, 1) - 1)/2);
    y = padarray(y, [1 1]*kernel_pad, 'replicate', 'both');
    for i = 1 : 4
        y = edgetaper(y, kernel);
    end
    [m, n, c] = size(y);
    for i = 1 : c
        x(:,:,i) = fast_deconv(y(:,:,i), kernel, lambda, alpha);
    end
    x = x(kernel_pad + 1 : m - kernel_pad, kernel_pad + 1 : n - kernel_pad,:);
    
end

