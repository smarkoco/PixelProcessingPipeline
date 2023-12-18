function rez = Kilosort_run_myo_3(ops_input_params, worker_id, worker_dir)
    script_dir = pwd; % get directory where repo exists
    load(fullfile(script_dir, '/tmp/config.mat'))
    load(fullfile(myo_sorted_dir, 'brokenChan.mat'))

    if num_KS_jobs > 1
        myo_sorted_dir = [myo_sorted_dir num2str(worker_id)];
    else
        dbstop if error % stop if error, if only one job
    end

    % get and set channel map
    if ~isempty(brokenChan) && remove_bad_myo_chans(1) ~= false
        chanMapFile = fullfile(myo_sorted_dir, 'chanMapAdjusted.mat');
    else
        chanMapFile = myo_chan_map_file;
    end
    disp(['Using this channel map: ' chanMapFile])

    try
        restoredefaultpath
    end

    addpath(genpath([script_dir '/sorting/Kilosort']))
    addpath(genpath([script_dir '/sorting/npy-matlab']))

    run([script_dir '/sorting/Kilosort_config_3.m']);
    ops.saveDir = myo_sorted_dir;
    ops.fbinary = fullfile(myo_sorted_dir, 'data.bin');
    ops.fproc = fullfile(myo_sorted_dir, 'proc.dat');
    ops.brokenChan = fullfile(myo_sorted_dir, 'brokenChan.mat');
    ops.chanMap = fullfile(chanMapFile);
    ops.nt0 = 61;
    ops.ntbuff = 64; % defined as 64;
    ops.NT = 2048 * 32 + ops.ntbuff; % convert to 32 count increments of samples % defined as 2048 * 32 + ops.ntbuff;
    ops.sigmaMask = Inf; % we don't want a distance-dependant decay
    ops.nPCs = 6; % how many PCs to project the spikes into (also used as number of template prototypes)
    ops.nEig = 3; % ops.nPCs; % rank of svd for templates, % keep same as nPCs to avoid error
    ops.NchanTOT = double(max(num_chans - length(brokenChan), ops.nEig)); %double(num_chans); %double(max(num_chans - length(brokenChan), 9));
    ops.Th = [10 2]; % threshold crossings for pre-clustering (in PCA projection space)
    ops.spkTh = -6; % spike threshold in standard deviations (-6 default) (only used in isolated_peaks_new)
    ops.nfilt_factor = 12; % max number of clusters per good channel (even temporary ones)
    ops.nblocks = 0;
    ops.nt0min = ceil(ops.nt0 / 2); % peak of template match will be this many points away from beginning
    ops.nskip = 25; % how many batches to skip for determining spike PCs
    ops.nSkipCov = 25; % compute whitening matrix and prototype templates using every N-th batch
    ops.lam = 10; % amplitude penalty (0 means not used, 10 is average, 50 is a lot)
    ops.CAR = 0; % whether to perform CAR
    ops.loc_range = [5 4]; % [timepoints channels], area to detect peaks; plus/minus for both time and channel. Doing abs() of data during peak isolation, so using 4 instead of default 5. Only 1 channel to avoid elimination of waves
    ops.long_range = [ops.nt0min-1 6]; % [timepoints channels], range within to use only the largest peak
    ops.fig = 1; % whether to plot figures
    ops.recordings = recordings;

    %% gridsearch section
    % only try to use gridsearch values if ops_input_params is a struct and fields are present
    if isa(ops_input_params, 'struct') && ~isempty(fieldnames(ops_input_params))
        % Combine input ops into the existing ops struct
        fields = fieldnames(ops_input_params);
        for iField = 1:size(fields, 1)
            ops.(fields{iField}) = ops_input_params.(fields{iField});
        end
        % ops.NT = ops.nt0 * 32 + ops.ntbuff; % 2*87040 % 1024*(32+ops.ntbuff);
    end
    %% end gridsearch section

    disp(['Using ' ops.fbinary])

    if trange(2) == 0
        ops.trange = [0 Inf];
    else
        ops.trange = trange;
    end

    % create parallel pool for all downstream parallel processing
    pc = parcluster('local');
    pc.JobStorageLocation = worker_dir;
    % ensure the number of processes across all workers does not exceed number of CPU cores
    % num_processes = 2*round(feature('numcores')/num_KS_jobs);
    % poolobj = parpool(pc, num_processes);
    poolobj = parpool(pc); % let matlab decide how many workers to use
    % ensure all parallel workers queues are cleared in the event of an error
    cleanup_worker_obj = onCleanup(@() cleanup_worker(poolobj));

    rez = preprocessDataSub(ops);
    % ops.channelDelays = rez.ops.channelDelays;
    rez = datashift2(rez, 1);
    [rez, st3, tF] = extract_spikes(rez);
if ops.fig
    %%% plots
    figure(5);
    plot(st3(:, 1), '.')
    title('Spike times versus spike ID')
    figure(6);
    plot(st3(:, 2), '.')
    title('Upsampled grid location of best template match spike ID')
    figure(7);
    plot(st3(:, 3), '.')
    title('Amplitude of template match for each spike ID')
    figure(8); hold on;
    plot(st3(:, 4), 'g.')
    for kSpatialDecay = 1:6
    less_than_idx = find(st3(:, 4) < ops.nPCs * kSpatialDecay);
    more_than_idx = find(st3(:, 4) >= ops.nPCs * (kSpatialDecay - 1));
    idx = intersect(less_than_idx, more_than_idx);
    bit_idx = bitand(st3(:, 4) < ops.nPCs * kSpatialDecay, st3(:, 4) >= ops.nPCs * (kSpatialDecay - 1));
    plot(idx, st3(bit_idx, 4), '.')
    end
    title('Prototype templates for each spatial decay value (1:6:30) resulting in each best match spike ID')
    figure(9);
    plot(st3(:, 5), '.')
    title('Amplitude of template match for each spike ID (Duplicate of st3(:,3))')
    figure(10);
    plot(st3(:, 6), '.')
    title('Batch ID versus spike ID')
    figure(11);
    for iPC = 1:size(tF, 2)
    subplot(size(tF, 2), 1, iPC)
            plot(squeeze(tF(:, iPC, :)), '.')
    end
title('PC Weights for each Spike Example, Colored by Channel')
        xlabel('Spike Examples')
        ylabel('Principal Component Weight')
    %%% end plots
end
    [rez] = template_learning(rez, tF, st3);
    [rez, st3, tF] = trackAndSort(rez);
    % plot_templates_on_raw_data_fast(rez, st3);
    rez = final_clustering(rez, tF, st3);
    rez = find_merges(rez, 1);

    % write to Phy
    disp(['Saving sorting results to Phy in', ops.saveDir])
    rezToPhy2(rez, ops.saveDir);

    disp(['Saving rez and ops structs to', ops.saveDir])
    ops % show final ops struct in command window
    rez % show final rez struct in command window

    % save variables as full struct, for MATLAB
    save(fullfile(ops.saveDir, '/ops_struct.mat'), 'ops');
    save(fullfile(ops.saveDir, '/rez_struct.mat'), 'rez');

    % save variables without struct, for python
    save(fullfile(ops.saveDir, '/ops.mat'), '-struct', 'ops');
rez.ops = []; rez.temp = []; % remove substructs from rez struct before saving
    save(fullfile(ops.saveDir, '/rez.mat'), '-struct', 'rez');
end

% cleanup function to ensure all parallel workers queues are cleared
function cleanup_worker(poolobj)
    % check if parallel pool processes exist
    if ~isempty(poolobj)
        delete(poolobj)
    end
    quit; % exit matlab to return to python
end
