function [wTEMP, wPCA] = extractTemplatesfromSnippets(rez, nPCs)
    % this function is very similar to extractPCfromSnippets.
    % outputs not just the PC waveforms, but also the template "prototype", 
    % basically k-means clustering of 1D waveforms. 
    
ops = rez.ops;

% skip every this many batches
nskip = getOr(ops, 'nskip', 25);

Nbatch      = rez.temp.Nbatch;
NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

fid = fopen(ops.fproc, 'r'); % open the preprocessed data file

k = 0;
dd = gpuArray.zeros(ops.nt0, 5e4, 'single'); % preallocate matrix to hold 1D spike snippets
if ops.fig == 1
    figure(1); hold on;
end
for ibatch = 1:nskip:Nbatch
    offset = 2 * ops.Nchan*batchstart(ibatch);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [ops.Nchan NT], '*int16');
    dat = dat';

    % move data to GPU and scale it back to unit variance
    dataRAW = gpuArray(dat);
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;


    % find isolated spikes from each batch
    [row, col, mu] = isolated_peaks_new(-abs(dataRAW), ops);

    % for each peak, get the voltage snippet from that channel
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);
    c = sq(clips(:, :));
    if ops.fig == 1
        plot(c)
    end
    if k+size(c,2)>size(dd,2)
        dd(:, 2*size(dd,2)) = 0;
    end
    
    dd(:, k + [1:size(c,2)]) = c;
    k = k + size(c,2);
    if k>1e5
        break;
    end
end
fclose(fid);
if ops.fig == 1
    title('local isolated spikes (1D voltage waveforms)');
end
% discard empty samples
dd = dd(:, 1:k);

% initialize the template clustering with random waveforms
% wTEMP = dd(:, randperm(size(dd,2), nPCs));
wTEMP = dd(:, round(linspace(1, size(dd,2), nPCs)));
wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % normalize them
if ops.fig == 1
    figure(2); hold on;
    for i = 1:nPCs
        plot(wTEMP(:,i)+i*1);
    end
    title('initial templates');
end
for i = 1:10
  % at each iteration, assign the waveform to its most correlated cluster
   cc = wTEMP' * dd;
   [amax, imax] = max(cc,[],1);
   for j = 1:nPCs
      wTEMP(:,j)  = dd(:,imax==j) * amax(imax==j)'; % weighted average to get new cluster means
   end
   wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % unit normalize
end
if ops.fig == 1
    figure(3); hold on;
    for i = 1:nPCs
        plot(wTEMP(:,i)+i*1);
    end
    title('prototype templates');
end

dd = double(gather(dd));
[U Sv V] = svdecon(dd); % the PCs are just the left singular vectors of the waveforms
if ops.fig == 1
    figure(4); hold on;
    for i = 1:nPCs
        plot(U(:,i)+i*1);
    end
    title(strcat("Top ", num2str(nPCs), " PCs"));
end
wPCA = gpuArray(single(U(:, 1:nPCs))); % take as many as needed
% adjust the arbitrary sign of the first PC so its peak is downward
wPCA(:,1) = - wPCA(:,1) * sign(wPCA(ops.nt0min,1));
