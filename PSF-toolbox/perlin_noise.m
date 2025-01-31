%Perlin noise simulation code, from "https://blog.csdn.net/weixin_42943114/article/details/82110468"
function bg_perlin = perlin_noise(imsz)
    zmat=zeros(imsz,imsz);
    for ii = 1:floor(log2(imsz))-1
        x_num_grid=2^ii;
        y_num_grid=2^ii;
        w = rand(1);
        zmat = zmat+ w*perlinnoise_generate(imsz,imsz,x_num_grid, y_num_grid);
    end
    bg_perlin = (zmat-min(zmat(:)))/(max(zmat(:))-min(zmat(:))); % bg_perlin = zmat/max(zmat(:));
end

%%
function output = perlinnoise_generate(x_num_pixel,y_num_pixel,x_num_grid, y_num_grid)
    x_pixel_per_grid = x_num_pixel/x_num_grid;
    y_pixel_per_grid = y_num_pixel/y_num_grid;
    limx=[0 x_num_grid];
    limy=[0 y_num_grid];
    dx=1/x_pixel_per_grid;
    dy=1/y_pixel_per_grid;
    zmat=perlinnoise2f(limx,limy,dx,dy);
    output = zmat(1:x_num_pixel, 1:y_num_pixel);
end

function zmat=perlinnoise2f(limx,limy,dx,dy)
    minx=limx(1);maxx=limx(2);
    miny=limy(1);maxy=limy(2);
    numx=maxx-minx+1;
    numy=maxy-miny+1;
    numx_z=(maxx-minx)/dx+1;
    numy_z=(maxy-miny)/dy+1;
    
    [uxmat,uymat]=randxymat(numx+1,numy+1);
    zmat=zeros(numy_z,numx_z);
    j=0;
    for x=minx:dx:maxx
        j=j+1;
        k=0;
        for y=miny:dy:maxy
            k=k+1;
            ddx=x-floor(x);
            ddy=y-floor(y);
            ux=uxmat(floor(y)-miny+1,floor(x)-minx+1);
            uy=uymat(floor(y)-miny+1,floor(x)-minx+1);
            n00=GridGradient(ux,uy,ddx,ddy);

            ddx=x-floor(x)-1;
            ddy=y-floor(y);
            ux=uxmat(floor(y)-miny+1,floor(x)-minx+2);
            uy=uymat(floor(y)-miny+1,floor(x)-minx+2);
            n10=GridGradient(ux,uy,ddx,ddy);

            ddx=x-floor(x);
            ddy=y-floor(y)-1;
            ux=uxmat(floor(y)-miny+2,floor(x)-minx+1);
            uy=uymat(floor(y)-miny+2,floor(x)-minx+1);
            n01=GridGradient(ux,uy,ddx,ddy);

            ddx=x-floor(x)-1;
            ddy=y-floor(y)-1;
            ux=uxmat(floor(y)-miny+2,floor(x)-minx+2);
            uy=uymat(floor(y)-miny+2,floor(x)-minx+2);
            n11=GridGradient(ux,uy,ddx,ddy);
            
            n0=lerp(n00,n10,x-floor(x));
            n1=lerp(n01,n11,x-floor(x));
            zmat(k,j)=lerp(n0,n1,y-floor(y));
        end
    end
    zmat=zmat+0.5;
    zmat(zmat>1)=1;zmat(zmat<0)=0;
end

function u=GridGradient(ux,uy,dx,dy)%
    u=ux*dx+uy*dy;
%     disp([ux,uy,dx,dy]);
end

function u=lerp(a,b,t)
    tw=6*t.^5-15*t.^4+10*t.^3;%6*t.^5-15*t.^4+10*t.^3;%3*t.^2-2*t.^3;%
    u=(1-tw)*a + tw*b;
end

function [uxmat,uymat]=randxymat(numx,numy)
    num=numx*numy;
    uxmat=zeros(numy,numx);
    uymat=zeros(numy,numx);
    for j=1:num
        k=0;
        while k==0
            randxy=rand(1,2);
            if (randxy(1)-0.5)^2+(randxy(2)-0.5)^2<=0.25
                k=k+1;
                uxmat(j)=randxy(1);
                uymat(j)=randxy(2);
            end
        end
    end
    uxmat=(uxmat-0.5)*2;
    uymat=(uymat-0.5)*2;
end
