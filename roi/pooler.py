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
        DYNAMIC_REVERSE = 'dynamic_reverse'
        DYNAMIC_128x128 = 'dynamic_128x128'
        DYNAMIC_256x256 = 'dynamic_256x256'
        DYNAMIC_512x512 = 'dynamic_512x512'

    OPTIONS = ['pooling', 'align', 'dynamic', 'dynamic_reverse', 'dynamic_128x128', 'dynamic_256x256', 'dynamic_512x512']

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
        elif mode in [Pooler.Mode.DYNAMIC, Pooler.Mode.DYNAMIC_REVERSE, Pooler.Mode.DYNAMIC_128x128, \
                      Pooler.Mode.DYNAMIC_256x256, Pooler.Mode.DYNAMIC_512x512]:
            pool = []
            if mode in [Pooler.Mode.DYNAMIC, Pooler.Mode.DYNAMIC_REVERSE]:
                sizes_sum = [0,0]
                counts = [0,0]
            # calculate it once for speed up inference
            _128x128 = 128*128
            _256x256 = 256*256
            _512x512 = 512*512
            for (proposal_bbox, proposal_batch_index) in zip(proposal_bboxes, proposal_batch_indices):
                bbox_square = proposal_bbox[2] * proposal_bbox[3]
                # iteratively calculating average online for maximize speed of forward pass
                if mode in [Pooler.Mode.DYNAMIC, Pooler.Mode.DYNAMIC_REVERSE]:
                    sizes_sum[0] += proposal_bbox[2]
                    counts[0] += 1
                    sizes_sum[1] += proposal_bbox[3]
                    counts[1] += 1
                    avg_square = (sizes_sum[0] / counts[0]) * (sizes_sum[1] / counts[1])
                    if mode == Pooler.Mode.DYNAMIC:
                        # proposal_bbox shape is x_shift, y_shift, weight, height
                        condition = bbox_square > avg_square
                    elif mode == Pooler.Mode.DYNAMIC_REVERSE:
                        condition = bbox_square < avg_square
                elif mode == Pooler.Mode.DYNAMIC_128x128:
                    condition = bbox_square > _128x128
                elif mode == Pooler.Mode.DYNAMIC_256x256:
                    condition = bbox_square > _256x256
                elif mode == Pooler.Mode.DYNAMIC_512x512:
                    condition = bbox_square > _512x512

                if condition:
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
                    features_cur_iter = features[proposal_batch_index].view(1, 256, features.shape[2], features.shape[3])
                    proposal_bbox_tensor = torch.stack(
                        [torch.Tensor([0, proposal_bbox[0], proposal_bbox[1], proposal_bbox[2], proposal_bbox[3]])],
                        dim=0).cuda()
                    out = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0)(
                        features_cur_iter,
                        proposal_bbox_tensor
                        ).view(256, 14, 14)
                pool.append(out)
            pool = torch.stack(pool, dim=0)
        else:
            raise ValueError

        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)
        return pool

