import torch


def get_encoder(cfg):
    if cfg.MODEL.ENCODER.lower() == 'swin':
        from model.encoder.swin import SwinTransformer
        backbone = SwinTransformer(img_size=224, embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=7)
        pretrained_dict = torch.load(cfg.TRAIN.PRETRAINED_SWIN_PATH)["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [128, 256, 512, 1024]
    elif cfg.MODEL.ENCODER.lower() == 'resnet':
        from model.encoder.resnet import ResNet50Backbone
        backbone = ResNet50Backbone()
        channel_list = [256, 512, 1024, 2048]
    elif cfg.MODEL.ENCODER.lower() == 'pvt':
        from model.encoder.pvt import pvt_v2_b5
        backbone = pvt_v2_b5()
        channel_list = [64, 128, 320, 512]
        # pvt_model_dict = backbone.state_dict()
        # pretrained_state_dicts = torch.load(cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        # state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        # pvt_model_dict.update(state_dict)
        # backbone.load_state_dict(pvt_model_dict)

    return backbone, channel_list
