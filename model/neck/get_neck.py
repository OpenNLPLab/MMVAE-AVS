from model.neck.neck import aspp_neck, basic_neck

def get_neck(cfg, in_channel_list):
    if cfg.MODEL.NECK.lower() == "aspp":
        neck = aspp_neck(in_channel_list=in_channel_list, out_channel=cfg.MODEL.NECK_CHANNEL)
    elif cfg.MODEL.NECK.lower() == "basic":
        neck = basic_neck(in_channel_list=in_channel_list, out_channel=cfg.MODEL.NECK_CHANNEL)
        
    return neck
