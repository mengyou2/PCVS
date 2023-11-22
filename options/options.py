


def get_model(opt):
    print("Loading model %s ... ")
    if opt.model_type == "zbuffer_pts":
        from models.z_buffermodel import ZbufferModelPts

        model = ZbufferModelPts(opt)
    elif opt.model_type == "z_buffermodel_w_depth_estimation":
        from models.z_buffermodel_w_depth_estimation import ZbufferModelPts
        model = ZbufferModelPts(opt)


    elif opt.model_type == "z_buffermodel_w_hfrefine":
        from models.z_buffermodel_hfrefine import ZbufferModelPts
        model = ZbufferModelPts(opt)


    elif opt.model_type == "z_buffermodel_w_depth_estimation_w_hfrefine":
        from models.z_buffermodel_w_depth_estimation_hfrefine import ZbufferModelPts

        model = ZbufferModelPts(opt)

        
    return model


def get_dataset(opt):

    print("Loading dataset %s ..." % opt.dataset)
    if opt.dataset == 'realestate10k':
        opt.min_z = 0.1
        opt.max_z = 10.0
        opt.train_data_path = (
            '/home/youmeng/data/RealEstate10K/'
        )
        from data.realestate10k import RealEstate10KDataLoader

        return RealEstate10KDataLoader
    elif opt.dataset == 'dtu':
        opt.min_z = 425
        opt.max_z = 900
        opt.train_data_path = (
            '/home/youmeng/data/DTU/'
        )
        from data.dtu import DTUDataLoader

        return DTUDataLoader


    elif opt.dataset == 'tanksandtemples':
        opt.min_z = 0.1
        opt.max_z = 10.0

        opt.train_data_path = (
            '/home/youmeng/data/tanks_and_temples/training/'
        )
        opt.eval_data_path = (
            '/home/youmeng/data/tanks_and_temples/testing/')
     
        from data.tanksandtemples import TanksandTemplesDataLoader

        return TanksandTemplesDataLoader


