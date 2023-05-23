% Copyright 2023 by Hui Yu & Yiming Wang
% Matlab version was Created by Yiming Wang
% If you have any comments, please feel free to contact Hui Yu or Yiming Wang.
% hui.yu@port.ac.uk  or  yiming.wang@port.ac.uk

function [ box ] = face_detect(data_folder, save_path, y_cor, x_cor)
% This function use the depth map to detect facial region
% box = face_detect('expdata\0002','recognition',[290 500],[250 440]);

if nargin<2
    save_path = [];
    y_cor = 0; x_cor = 0;
end
[f_align,d_align,ar_align] = syncro(data_folder);
load([data_folder '\ind.mat'],'facedata')
load([data_folder '\dep.mat'],'dep')
height = size(dep(1).data,1);
im_num = length(f_align);
box = zeros(im_num,4);
for i = 1:im_num
    d_map = dep(f_align(i)).data;
    d_map(d_map<60)=0;
    c=diff((d_map==0)')';
    seq_zero_num = ones(height,2)*640;
    for j = 1:height
        if ~isempty(find(c(j,:)==1, 1))
            diff_ind = find(c(j,:)~=0);
            to_zero_pos = find(c(j,:)==1);
            check_ind = find(to_zero_pos>y_cor(1),1);
            if isempty(check_ind)
                continue;
            end
            start_ind = find(diff_ind==to_zero_pos(check_ind),1);
            end_ind = find(diff_ind>y_cor(2),1);
            if isempty(end_ind)
                end_ind = length(diff_ind);
            end
            diff_ind=diff_ind(start_ind:end_ind);
            d=diff(diff_ind);
            if isempty(d)
                continue;
            end
            [max_d,max_ind] = max(d(1:2:end));
            seq_zero_num(j,:)=[diff_ind(max_ind*2-1) max_d];
        end
    end
    to_remove_invalid = seq_zero_num(:,2)<110;
    seq_zero_num(to_remove_invalid,2)=640;
    [~,neck_ind] = min(seq_zero_num(x_cor(1)+1:x_cor(2),2));
    box(i,:) = [seq_zero_num(neck_ind+x_cor(1),1) neck_ind+x_cor(1)-seq_zero_num(neck_ind+x_cor(1),2) seq_zero_num(neck_ind+x_cor(1),2) seq_zero_num(neck_ind+x_cor(1),2)];
end
% box_diff=diff(box(:,1)); mis_det=find(abs(box_diff)>30);
% cor = mis_det(1:2:end); incor = mis_det(2:2:end); box(incor,:)=box(cor,:);
incor_ind = find(abs(box(:,1)-box(1,1))>70);
for i = 1:length(incor_ind)
    box(incor_ind(i),:) = box(incor_ind(i)-1,:);
end
if ~isempty(save_path)
    box(:,1)=box(:,1)-20;
    box(:,2)=box(:,2)-10;
    box(:,3)=box(:,3)+box(:,1);
    box(:,4)=box(:,4)+box(:,2);
    for i = 1:im_num
        strI = strfind(facedata(f_align(i)).name,data_folder);
        im_path = facedata(f_align(i)).name(strI:end);
        im = imread(im_path);
        croped_im = imresize(im(box(i,2):box(i,4),box(i,1):box(i,3),:),[224,224]);
        out_path = [save_path '\' data_folder];
        if exist(out_path,'dir')==0
            mkdir(out_path);
        end
        out_imname = [out_path '\' num2str(i,'%05d') '.png'];
        imwrite(croped_im,out_imname)
    end
end