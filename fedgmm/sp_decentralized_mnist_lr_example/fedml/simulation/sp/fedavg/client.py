from torch.optim.lr_scheduler import CosineAnnealingLR
class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)
        # self.args.epochs = epochs 

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, g_global, f_global):
        self.model_trainer.set_g_model_params(g_global)
        self.model_trainer.set_f_model_params(f_global)
        self.model_trainer.train_gmm(self.local_training_data, self.device, self.args)
        g = self.model_trainer.get_g_model_params()
        f = self.model_trainer.get_f_model_params()
        return [g, f]
    
    def train_reg(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
