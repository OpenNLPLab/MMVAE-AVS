import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



def structure_loss_ms3(pred_masks, mask):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred_masks, new_gts, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_masks = torch.sigmoid(pred_masks)
    inter = ((pred_masks * mask) * weit).sum(dim=(2, 3))
    union = ((pred_masks + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def structure_loss_s4(pred_masks, mask):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts

    indices = torch.tensor(list(range(0, len(pred_masks), 5)), device=pred_masks.device)
    pred = torch.index_select(pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
    
    if len(mask.shape) == 5:
        mask = mask.squeeze(1) # [bs, 1, 224, 224]
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks) # [bs*5, 1, 224, 224]
    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()

    first_pred = torch.index_select(pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1) # [bs, 1, 224, 224]
    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss


def A_MaskedV_SimmLoss(pred_masks, a_fea_list, v_map_list, mask_pooling_type='avg', norm_fea=True, threshold=False, euclidean_flag=False, kl_flag=False):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_AV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    total_loss = 0
    for stage in range(len(a_fea_list)):
        a_fea, v_map = a_fea_list[stage], v_map_list[stage] # v_map: [BT, C, H, W]
        BT, C, H, W = v_map.shape
        a_fea = a_fea.permute(0, 2, 1).view(BT, C) # [B*5, C]

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        downsample_pred_masks = torch.sigmoid(downsample_pred_masks) # [B*5, 1, H, W]

        if threshold:
            downsample_pred_masks = (downsample_pred_masks > 0.5).float() # [bs*5, 1, H, W]
            obj_pixel_num = downsample_pred_masks.sum(-1).sum(-1) # [bs*5, 1]
            masked_v_map = torch.mul(v_map, downsample_pred_masks)  # [bs*5, C, H, W]
            masked_v_fea = masked_v_map.sum(-1).sum(-1) / (obj_pixel_num + 1e-6)# [bs*5, C]
        else:
            masked_v_map = torch.mul(v_map, downsample_pred_masks)
            masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*5, C]

        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        if euclidean_flag:
            euclidean_distance = F.pairwise_distance(a_fea, masked_v_fea, p=2)
            loss = euclidean_distance.mean()
        elif kl_flag:
            loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='sum')
        total_loss += loss

    total_loss /= len(a_fea_list)

    return total_loss


def IouSemanticAwareLoss(pred_masks, first_gt_mask, a_fea_list, v_map_list, 
                         lambda_1=0, mask_pooling_type='avg'):
    """
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    a_fea_list: feature list of audio features
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W]
    """
    total_loss = 0
    f1_iou_loss = structure_loss_s4(pred_masks, first_gt_mask)
    total_loss += f1_iou_loss

    if lambda_1 > 0.0:
        sa_loss = A_MaskedV_SimmLoss(pred_masks, a_fea_list, v_map_list, mask_pooling_type, kl_flag=True)
        total_loss += lambda_1 * sa_loss
    else:
        sa_loss = torch.zeros(1)

    # pdb.set_trace()
    loss_dict = {}
    loss_dict['iou_loss'] = f1_iou_loss.item()
    loss_dict['sa_loss'] = sa_loss.item()
    loss_dict['lambda_1'] = lambda_1
    return total_loss, loss_dict


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
