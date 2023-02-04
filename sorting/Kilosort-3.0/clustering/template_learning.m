function rez  = template_learning(rez, tF, st3)

wPCA  = rez.wPCA;
iC = rez.iC;
ops = rez.ops;


xcup = rez.xcup;
ycup = rez.ycup;

wroll = [];
tlag = [-2, -1, 1, 2];
for j = 1:length(tlag)
    wroll(:,:,j) = circshift(wPCA, tlag(j), 1)' * wPCA;
end

%% split templates into batches
rmin = 0.6;
nlow = 100;
n0 = 0;
use_CCG = 0;

Nchan = rez.ops.Nchan;
Nk = size(iC,2);
yunq = unique(rez.yc);

ktid = int32(st3(:,2)) + 1;
tmp_chan = iC(1, :);
ss = double(st3(:,1)) / ops.fs;

dmin = rez.ops.dmin;
ycenter = (min(rez.yc) + dmin-1):(2*dmin):(max(rez.yc)+dmin+1);
dminx = rez.ops.dminx;
xcenter = (min(rez.xc) + dminx-1):(2*dminx):(max(rez.xc)+dminx+1);
[xcenter, ycenter] = meshgrid(xcenter, ycenter);
xcenter = xcenter(:);
ycenter = ycenter(:);

Wpca = zeros(size(tF,2), Nchan, 1000, 'single');
nst = numel(ktid);
hid = zeros(nst,1 , 'int32');

% ycup = rez.yc;


tic

for j = 1:numel(ycenter)
    if rem(j,5)==1
        fprintf('time %2.2f, GROUP %d/%d, units %d \n', toc, j, numel(ycenter), n0)    
    end
    
    y0 = ycenter(j);
    x0 = xcenter(j);    
    xchan = (abs(ycup - y0) < dmin) & (abs(xcup - x0) < dminx);
    itemp = find(xchan);
        
    if isempty(itemp)
        continue;
    end
    tin = ismember(ktid, itemp);
    pid = ktid(tin);
    data = tF(tin, :, :);
    
    if isempty(data)
        continue;
    end
%     size(data)
    
    %https://github.com/MouseLand/Kilosort/issues/427
    try
        ich = unique(iC(:, itemp));
    catch
        tmpS = iC(:, itemp);
        ich = unique(tmpS);
    end
%     ch_min = ich(1)-1;
%     ch_max = ich(end);
    
    if numel(ich)<1
        continue;
    end
    
    nsp = size(data,1);
    dd = zeros(nsp, size(data,2), numel(ich), 'single');
    for k = 1:length(itemp)
        ix = pid==itemp(k);
        % how to go from channels to different order
        [~,ia,ib] = intersect(iC(:,itemp(k)), ich);
        dd(ix, :, ib) = data(ix,:,ia);
    end

    kid = run_pursuit(dd, nlow, rmin, n0, wroll, ss(tin), use_CCG);

    [~, ~, kid] = unique(kid);
    nmax = max(kid);
    for t = 1:nmax
%         Wpca(:, ch_min+1:ch_max, t + n0) = gather(sq(mean(dd(kid==t,:,:),1)));
        Wpca(:, ich, t + n0) = gather(sq(mean(dd(kid==t,:,:),1)));
    end
    
    hid(tin) = gather(kid + n0);
    n0 = n0 + nmax;
end
Wpca = Wpca(:,:,1:n0);
toc
%%
sdfdsfdf
rez.W = zeros(ops.nt0, 0, ops.nEig, 'single');
rez.U = zeros(ops.Nchan, 0, ops.nEig, 'single');
rez.mu = zeros(1,0, 'single');
for  t = 1:n0
    dWU = wPCA * gpuArray(Wpca(:,:,t));
    [w,s,u] = svdecon(dWU);
    wsign = -sign(w(ops.nt0min,1)); %% time was hard-coded...
    rez.W(:,t,:) = gather(wsign * w(:,1:ops.nEig));
    rez.U(:,t,:) = gather(wsign * u(:,1:ops.nEig) * s(1:ops.nEig,1:ops.nEig));
    rez.mu(t) = gather(sum(sum(rez.U(:,t,:).^2))^.5);
    rez.U(:,t,:) = rez.U(:,t,:) / rez.mu(t);
end
%%
rez.ops.wPCA = wPCA;

