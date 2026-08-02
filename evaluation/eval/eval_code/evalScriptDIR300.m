function [mean_ms, mean_ld, mean_ad] = evalScriptDIR300(gtdir_dir300, imdir_dir300, verbose)
% add LD path
% https://people.csail.mit.edu/celiu/SIFTflow/
% change the path to your SIFTflow folder
addpath(genpath('./SIFTflow'));

res = cell(300, 1);

parfor k = 1 : 300
    if ~isfile(sprintf('%s/%d.png', gtdir_dir300, k))
        fprintf('%s - Not file (ref/input)\n', sprintf('%s/%d.png', gtdir_dir300, k))
        res{k + 1} =[k, -1, -1, -1];
        continue 
    end
    rimg = rgb2gray(imread(sprintf('%s/%d.png', gtdir_dir300, k)));
    t = zeros(4);
    if isfile(sprintf('%s/%d.png', imdir_dir300, k))
        if verbose
            fprintf('Running %d ... ', k)
        end
        ximg = rgb2gray(imread(sprintf('%s/%d.png', imdir_dir300, k)));
        [rh,rw,~]=size(rimg);
        rimg=imresize(rimg,sqrt(598400/(rh*rw)),'bicubic');
        [rh,rw,~]=size(rimg);
        ximg=imresize(ximg,[rh rw],'bicubic');
        
        [vx, vy] = siftFlow(rimg, ximg);
        ad = evalAlignedUnwarp(ximg, rimg, vx, vy);

        [ms, ld] = evalUnwarp(ximg, rimg, vx, vy);

        t = [k, ad, ms, ld];
    else
        t = [k, -1, -1, -1];
        fprintf('%s - Not file\n', sprintf('%s/%d.png', imdir_dir300, k))
    end

    res{k + 1} = t;
end
res = cell2mat(res);
valres = res(res(:, 3) > 0, :);
avg = mean(valres, 1);

mean_ms = avg(3); 
mean_ld = avg(4);
mean_ad = avg(2);

res = cat(1, res, avg);

save(sprintf('%s/individual_res.txt', imdir_dir300), 'res', '-ascii');
end