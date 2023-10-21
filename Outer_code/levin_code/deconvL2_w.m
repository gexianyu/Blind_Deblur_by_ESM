function [x]=deconvL2_w(I,k,we,max_it,weight_x,weight_y,weight_xx,weight_yy,weight_xy)

    if (~exist('max_it','var'))
       max_it=200;
    end

    [N1,N2]=size(I);

    [fs_y,fs_x]=size(k);
    hfs1_x1=floor((size(k,2)-1)/2);
    hfs1_x2=ceil((size(k,2)-1)/2);
    hfs1_y1=floor((size(k,1)-1)/2);
    hfs1_y2=ceil((size(k,1)-1)/2);
    shifts1=[-hfs1_x1  hfs1_x2  -hfs1_y1  hfs1_y2];

    hfs_x1=hfs1_x1;
    hfs_x2=hfs1_x2;
    hfs_y1=hfs1_y1;
    hfs_y2=hfs1_y2;


    N2=N2+hfs_x1+hfs_x2;
    N1=N1+hfs_y1+hfs_y2;
    N=N2*N1;
    mask=zeros(N1,N2);
    mask(hfs_y1+1:N1-hfs_y2,hfs_x1+1:N2-hfs_x2)=1;

    if (~exist('weight_x','var'))
      weight_x=ones(N1,N2-1);
      weight_y=ones(N1-1,N2);
      weight_xx=zeros(N1,N2-2);
      weight_yy=zeros(N1-2,N2);
      weight_xy=zeros(n-1,m-1);
    end


    tI=I;
    I=zeros(N1,N2);
    I(hfs_y1+1:N1-hfs_y2,hfs_x1+1:N2-hfs_x2)=tI; 
    x=tI([ones(1,hfs_y1),1:end,end*ones(1,hfs_y2)],[ones(1,hfs_x1),1:end,end*ones(1,hfs_x2)]);



    b=conv2(x.*mask,k,'same');

    %pad k with zeros up to a nearby integer with small prime factors, for fast fft
    N1p=goodfactor(N1+hfs1_y1+hfs1_y2);
    N2p=goodfactor(N2+hfs1_x1+hfs1_x2);
    K=zero_pad2(k,ceil((N1p-fs_y)/2),floor((N1p-fs_y)/2),ceil((N2p-fs_x)/2),floor((N2p-fs_x)/2));
    K=fft2(ifftshift(K));

    dxf=[1 -1];
    dyf=[1;-1];
    dyyf=[-1; 2; -1];
    dxxf=[-1, 2, -1];
    dxyf=[-1 1;1 -1];

    if (max(size(k)<=5))
      Ax=conv2(conv2(x,rot90(k,2),'same').*mask,  k,'same');
    else
      Ax=fftconvf(fftconvf(x,rot90(k,2),conj(K),'same').*mask,k,K,'same');
    end


    Ax=Ax+we*conv2(weight_x.*conv2(x,rot90(dxf,2),'valid'),dxf);
    Ax=Ax+we*conv2(weight_y.*conv2(x,rot90(dyf,2),'valid'),dyf);
    Ax=Ax+we*(conv2(weight_xx.*conv2(x,rot90(dxxf,2),'valid'),dxxf));
    Ax=Ax+we*(conv2(weight_yy.*conv2(x,rot90(dyyf,2),'valid'),dyyf));
    Ax=Ax+we*(conv2(weight_xy.*conv2(x,rot90(dxyf,2),'valid'),dxyf));


    r = b - Ax;

    for iter = 1:max_it  
         rho = (r(:)'*r(:));

         if ( iter > 1 )                       % direction vector
            beta = rho / rho_1;
            p = r + beta*p;
         else
            p = r;
         end
         if (max(size(k)<5))
           Ap=conv2(conv2(p,rot90(k,2),'same').*mask,  k,'same');
         else  
           Ap=fftconvf(fftconvf(p,rot90(k,2),conj(K),'same').*mask,k,K,'same');
         end

         Ap=Ap+we*conv2(weight_x.*conv2(p,rot90(dxf,2),'valid'),dxf);
         Ap=Ap+we*conv2(weight_y.*conv2(p,rot90(dyf,2),'valid'),dyf);
         Ap=Ap+we*(conv2(weight_xx.*conv2(p,rot90(dxxf,2),'valid'),dxxf));
         Ap=Ap+we*(conv2(weight_yy.*conv2(p,rot90(dyyf,2),'valid'),dyyf));
         Ap=Ap+we*(conv2(weight_xy.*conv2(p,rot90(dxyf,2),'valid'),dxyf));


         q = Ap;
         alpha = rho / (p(:)'*q(:) );
         x = x + alpha * p;                    % update approximation vector

         r = r - alpha*q;                      % compute residual

         rho_1 = rho;
    end
end


function N=goodfactor(N)
    
    f=factor(N);
    
    while(max(f)>7)
      N=N+1;
      f=factor(N);
    end
end
    
function zM=zero_pad2(M,zp1d,zp1u,zp2d,zp2u)

    [n,m,k]=size(M);
    zM=zeros(n+zp1u+zp1d,m+zp2d+zp2u,k);
    zM(zp1d+1:end-zp1u,zp2d+1:end-zp2u,:)=M;
end

function cI=fftconvf(I,k,K,method)

    [N1,N2]=size(I);
    [k1,k2]=size(k);
    hk1=(k1-1)/2;
    hk2=(k2-1)/2;
    [bk1,bk2]=size(K);
    hdiff1d=ceil((bk1-N1)/2);
    hdiff1u=floor((bk1-N1)/2);
    hdiff2d=ceil((bk2-N2)/2);
    hdiff2u=floor((bk2-N2)/2);

    I=zero_pad2(I,hdiff1d,hdiff1u,hdiff2d,hdiff2u);

    fI=fft2(ifftshift(I));
    cI=fftshift(ifft2(fI.*K));

    if exist('method','var')

        if strcmp(method,'same')
          cI=cI(hdiff1d+1:end-hdiff1u,hdiff2d+1:end-hdiff2u);    
        end
        if strcmp(method,'valid')
          cI=cI(hdiff1d+1:end-hdiff1u,hdiff2d+1:end-hdiff2u);    
          cI=cI(hk1+1:end-hk1,hk2+1:end-hk2);     
        end
    end
end
