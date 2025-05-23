import copy
import torch
from torch import nn
import torch
from torch.optim import Optimizer
from ...core.alg_frame.client_trainer import ClientTrainer
import logging
from optimizers import fedoptimizer

class FedProxModelTrainer(ClientTrainer):
    def get_g_model_params(self):
        return self.g.state_dict()
    def get_f_model_params(self):
        return self.f.state_dict()
    
    def get_model_params(self):
        return self.reg_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.reg_model.load_state_dict(model_parameters)
        self.reg_model = self.reg_model.train()  

    def set_g_model_params(self, model_parameters):
        self.g.load_state_dict(model_parameters)
        self.g = self.g.train()
    def set_f_model_params(self, model_parameters):
        self.f.load_state_dict(model_parameters)
        self.f = self.f.train()
    # def get_model_params(self):
    #     return self.model.cpu().state_dict()

    # def set_model_params(self, model_parameters):
    #     self.model.load_state_dict(model_parameters)

    def train(self, client_data, device, args):
        model = self.reg_model

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.reg_model.parameters()),
                lr=args.learning_rate,

            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        

        # epoch_loss = []
        # for epoch in range(args.epochs):
        #     batch_loss = []
        #     for batch_idx, (x, labels) in enumerate(train_data):
        #         x, labels = x.to(device), labels.to(device)
        #         model.zero_grad()
        #         log_probs = model(x)
        #         loss = criterion(log_probs, labels)  # pylint: disable=E1102
        #         # if args.fedprox:
        #         fed_prox_reg = 0.0
        #         for name, param in model.named_parameters():
        #             fed_prox_reg += ((args.fedprox_mu / 2) * \
        #                 torch.norm((param - previous_model[name].data.to(device)))**2)
        #         loss += fed_prox_reg

        #         loss.backward()

        #         # Uncommet this following line to avoid nan loss
        #         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        #         optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for epoch in range(args.epochs):
                for batch in client_data:
                    x_batch = batch[2]
                    y_batch = batch[3]
                    model.zero_grad()
                    preds = torch.squeeze(model(x_batch))
                    truee = torch.squeeze(y_batch)
                    loss = criterion(preds, truee)  # pylint: disable=E1102
                    fed_prox_reg = 0.0
                    for name, param in model.named_parameters():
                     fed_prox_reg += ((args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                    loss += fed_prox_reg
                    loss.backward()
                    optimizer.step()


                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )
        # model = self.reg_model
        # # model = model.load_state_dict(self.get_model_params())
        # model.to(device)
        # model.train()

        # # train and update
        # criterion = nn.MSELoss().to(device)  # pylint: disable=E1102
        # if args.client_optimizer == "sgd":
        #     optimizer = torch.optim.SGD(
        #         filter(lambda p: p.requires_grad, self.reg_model.parameters()),
        #         lr=args.learning_rate,
        #     )
        # else:
        #     optimizer = torch.optim.Adam(
        #         filter(lambda p: p.requires_grad, self.model.parameters()),
        #         lr=args.learning_rate,
        #         weight_decay=args.weight_decay,
        #         amsgrad=True,
        #     )

        # epoch_loss = []
        # for epoch in range(args.epochs):
        #     batch_loss = []

        #     for epoch in range(args.epochs):
        #         for batch in client_data:
        #             x_batch = batch[2]
        #             y_batch = batch[3]
        #             model.zero_grad()
        #             preds = torch.squeeze(model(x_batch))
        #             truee = torch.squeeze(y_batch)
        #             loss = criterion(preds, truee)  # pylint: disable=E1102
        #             loss.backward()
        #             optimizer.step()

        #         # Uncommet this following line to avoid nan loss
        #         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        #         # logging.info(
        #         #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #         #         epoch,
        #         #         (batch_idx + 1) * args.batch_size,
        #         #         len(train_data) * args.batch_size,
        #         #         100.0 * (batch_idx + 1) / len(train_data),
        #         #         loss.item(),
        #         #     )
        #         # )

        #         batch_loss.append(loss.item())
        #     if len(batch_loss) == 0:
        #         epoch_loss.append(0.0)
        #     else:
        #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
        #     # logging.info(
        #     #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
        #     #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
        #     #     )
        #     # )
    def train_gmm(self, client_data, device, args,additional_epochs):
        g = self.g
        f = self.f
        
        g.to(device)
        f.to(device)
        
        g.train()
        f.train()
        previous_g = copy.deepcopy(g.state_dict())
        previous_f = copy.deepcopy(f.state_dict())

        for epoch in range(args.epochs):
            for batch in client_data:
                x_batch = batch[2]
                y_batch = batch[3]
                z_batch = batch[4]
                g_obj, f_obj = self.game_objective.calc_objective(
                    g, f, x_batch, z_batch, y_batch)
                # do single step optimization on f and g
                fed_prox_reg = 0.0
                for name, param in g.named_parameters():
                    fed_prox_reg += ((args.fedprox_mu / 2) * \
                        torch.norm((param - previous_g[name].data.to(device)))**2)
                g_obj += fed_prox_reg

                fed_prox_reg = 0.0
                for name, param in f.named_parameters():
                    fed_prox_reg += ((args.fedprox_mu / 2) * \
                        torch.norm((param - previous_f[name].data.to(device)))**2)
                f_obj += fed_prox_reg

                self.g_optimizer.zero_grad()
                g_obj.backward(retain_graph=True)
                self.g_optimizer.step()

                self.f_optimizer.zero_grad()
                f_obj.backward()
                self.f_optimizer.step()
        
        # self.compare_params(initial_g_params, g, "g")
        # self.compare_params(initial_f_params, f, "f")
        
        self.set_g_model_params(g.state_dict())
        self.set_f_model_params(f.state_dict())

    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )








    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
