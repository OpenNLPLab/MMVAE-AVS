def get_decoder(cfg):
    if cfg.MODEL.DECODER.lower() == 'rcab':
        from model.decoder.rcab_decoder import rcab_decoder
        decoder = rcab_decoder(neck_channel=cfg.MODEL.NECK_CHANNEL)
    elif cfg.MODEL.DECODER.lower() == 'simple':
        from model.decoder.simple_decoder import simple_decoder
        decoder = simple_decoder(neck_channel=cfg.MODEL.NECK_CHANNEL)
    elif cfg.MODEL.DECODER.lower() == 'simple_no_module':
        from model.decoder.simple_decoder import simple_decoder_no_modulelist
        decoder = simple_decoder_no_modulelist(neck_channel=cfg.MODEL.NECK_CHANNEL)
    else:
        raise KeyError('No decoder named {}'.format(cfg.MODEL.DECODER))

    return decoder
