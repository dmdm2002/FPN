class Param(object):
    def __init__(self):
        super(Param, self).__init__()

        self.root = ''
        self.db_path = f'{self.root}/'
        self.output_ckp = f'{self.root}/backup/ckp'
        self.output_log = f'{self.root}/backup/log'

        self.device = "cpu"

        self.max_epoch = 100
        self.lr = 1e-4
        self.backbone = 'resnet152'
        self.batchsz = 2

        self.input_size = 224
        self.do_load_ckp = False
        self.load_epoch = 0
        self.run_type = ['train', 'test']
