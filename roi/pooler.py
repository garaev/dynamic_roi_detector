from enum import Enum

import torch
from torch import Tensor
from torch.nn import functional as F
#from torchvision.ops import roi_pool, roi_align

from support.layer.roi_align import ROIAlign


class Pooler(object):

    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'
        DYNAMIC = 'dynamic'
        ALIGN_2 = 'align_2'

    OPTIONS = ['pooling', 'align', 'dynamic', 'align_2']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor, proposal_batch_indices: Tensor, mode: Mode) -> Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        scale = 1 / 16
        output_size = (7 * 2, 7 * 2)

        if mode == Pooler.Mode.POOLING:
            pool = []
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                start_x = max(min(round(proposal_bbox[0].item() * scale), feature_map_width - 1), 0)      # [0, feature_map_width)
                start_y = max(min(round(proposal_bbox[1].item() * scale), feature_map_height - 1), 0)     # (0, feature_map_height]
                end_x = max(min(round(proposal_bbox[2].item() * scale) + 1, feature_map_width), 1)        # [0, feature_map_width)
                end_y = max(min(round(proposal_bbox[3].item() * scale) + 1, feature_map_height), 1)       # (0, feature_map_height]
                roi_feature_map = features[proposal_batch_index, :, start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=output_size))
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.ALIGN:
            pool = []
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                a = features[proposal_batch_index].view(1, 256, features.shape[2], features.shape[3])
                b = torch.stack([torch.Tensor([0, proposal_bbox[0], proposal_bbox[1], proposal_bbox[2], proposal_bbox[3]])], dim=0).cuda()
                x = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                    a,
                    b
                ).view(256,14,14)
                pool.append(x)
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.DYNAMIC:
            pool = []
            avg = [0,0]
            counts = [0,0]
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                #print(proposal_bbox[2], proposal_bbox[3])
                avg[0] += proposal_bbox[2]
                counts[0] += 1
                avg[1] += proposal_bbox[3]
                counts[1] += 1
                if proposal_bbox[2]*proposal_bbox[3] > 512*512: #(avg[0]/counts[0])*(avg[1]/counts[1])
                #if random.choice([0,1]):
                    start_x = max(min(round(proposal_bbox[0].item() * scale), feature_map_width - 1),
                                  0)  # [0, feature_map_width)
                    start_y = max(min(round(proposal_bbox[1].item() * scale), feature_map_height - 1),
                                  0)  # (0, feature_map_height]
                    end_x = max(min(round(proposal_bbox[2].item() * scale) + 1, feature_map_width),
                                1)  # [0, feature_map_width)
                    end_y = max(min(round(proposal_bbox[3].item() * scale) + 1, feature_map_height),
                                1)  # (0, feature_map_height]
                    roi_feature_map = features[proposal_batch_index, :, start_y:end_y, start_x:end_x]
                    out = F.adaptive_max_pool2d(input=roi_feature_map, output_size=output_size)
                else:
                    a = features[proposal_batch_index].view(1, 256, features.shape[2], features.shape[3])
                    b = torch.stack(
                        [torch.Tensor([0, proposal_bbox[0], proposal_bbox[1], proposal_bbox[2], proposal_bbox[3]])],
                        dim=0).cuda()
                    out = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                        a,
                        b
                        ).view(256, 14, 14)
                pool.append(out)
            #print((avg[0]/counts[0]), (avg[1]/counts[1]))
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.ALIGN_2:
            pool = []
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                a = features[proposal_batch_index].view(1, 256, features.shape[2], features.shape[3])
                b = torch.stack(
                    [torch.Tensor([0, proposal_bbox[0], proposal_bbox[1], proposal_bbox[2], proposal_bbox[3]])],
                    dim=0).cuda()
                out = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                    a,
                    b
                    ).view(256, 14, 14)
                pool.append(out)
            pool = torch.stack(pool, dim=0)
        else:
            raise ValueError

        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)
        return pool

