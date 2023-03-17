import os
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from meta_neural_network_architectures import VGGReLUNormNetwork, ResNet12
from inner_loop_optimizers import LSLRGradientDescentLearningRule
from scipy.special import softmax 
import random
def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def euclidean_dist(x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0
        self.rng = set_torch_seed(seed=args.seed)

        if self.args.backbone == 'ResNet12':
            self.classifier = ResNet12(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)
        else:
            self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)

        self.task_learning_rate = args.init_inner_loop_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    init_weight_decay=args.init_inner_loop_weight_decay,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_weight_decay=self.args.alfa,
                                                                    use_learnable_learning_rates=self.args.alfa,
                                                                    alfa=self.args.alfa, random_init=self.args.random_init)

        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        if self.args.attenuate:
            num_layers = len(names_weights_copy)
            self.attenuator = nn.Sequential(
                nn.Linear(num_layers, num_layers),
                nn.ReLU(inplace=True),
                nn.Linear(num_layers, num_layers),
                nn.Sigmoid()
            ).to(device=self.device)

        self.inner_loop_optimizer.initialise(
            names_weights_dict=names_weights_copy)

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        # ALFA
        if self.args.alfa:
            num_layers = len(names_weights_copy)
            input_dim = num_layers*2
            output_dim = num_layers
            self.regularizer1 = nn.LSTM(input_dim, output_dim,num_layers = 2)
            self.regularizer2 = nn.LSTM(input_dim, output_dim,num_layers = 2)


        if self.args.attenuate:
            if self.args.alfa:
                self.optimizer = optim.Adam([
                    {'params':self.classifier.parameters()},
                    {'params': self.inner_loop_optimizer.parameters()},
                    {'params': self.regularizer.parameters()},
                    {'params':self.attenuator.parameters()},
                ],lr=args.meta_learning_rate, amsgrad=False)
            else:
                self.optimizer = optim.Adam([
                    {'params':self.classifier.parameters()},
                    {'params':self.attenuator.parameters()},
                ],lr=args.meta_learning_rate, amsgrad=False)
        else:
            if self.args.alfa:
                if self.args.random_init:
                    self.optimizer = optim.Adam([
                        {'params': self.inner_loop_optimizer.parameters()},
                        {'params': self.regularizer1.parameters()},
                    ], lr=args.meta_learning_rate, amsgrad=False )
                else:
                    self.optimizer = optim.Adam([
                        {'params': self.classifier.parameters()},
                        {'params': self.inner_loop_optimizer.parameters()},
                        {'params': self.regularizer1.parameters()},
                        {'params': self.regularizer2.parameters()},
                    ], lr=args.meta_learning_rate, amsgrad=False)
            else:
                self.optimizer = optim.Adam([
                    {'params': self.classifier.parameters()},
                ], lr=args.meta_learning_rate, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()

    def get_task_embeddings(self, x_support_set_task, y_support_set_task, names_weights_copy):
        # Use gradients as task embeddings
        support_loss, support_preds, support_loss_s = self.net_forward(x=x_support_set_task,
                                                       y=y_support_set_task,
                                                       weights=names_weights_copy,
                                                       backup_running_statistics=True,
                                                       training=True, num_step=0)

        if torch.cuda.device_count() > 1:
            self.classifier.module.zero_grad(names_weights_copy)
        else:
            self.classifier.zero_grad(names_weights_copy)
        grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=True)


        layerwise_mean_grads = []

        for i in range(len(grads)):
            layerwise_mean_grads.append(grads[i].mean())

        layerwise_mean_grads = torch.stack(layerwise_mean_grads)

        return layerwise_mean_grads

    def attenuate_init(self, task_embeddings, names_weights_copy):
        # Generate attenuation parameters
        gamma = self.attenuator(task_embeddings)

        ## Attenuate

        updated_names_weights_copy = dict()
        i = 0
        for key in names_weights_copy.keys():
            updated_names_weights_copy[key] = gamma[i] * names_weights_copy[key]
            i+=1

        return updated_names_weights_copy


    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, generated_alpha_params, generated_beta_params, momentum, use_second_order, current_step_idx,support_data,support_label,num_step,learning_rate,loss_count):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)
          
        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True,retain_graph = True)

        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))                                                     
        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}
        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
        cos_sim = []
        if num_step > 0:
            for k in names_weights_copy:
                    learning_rate[k] = F.cosine_similarity(names_grads_copy[k].flatten(), momentum[k].flatten(), dim = 0)*2
                    cos_sim.append(learning_rate[k].item()/2)
            names_weights_grad = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                        names_grads_wrt_params_dict=names_grads_copy,
                                                                        generated_alpha_params=generated_alpha_params,
                                                                        generated_beta_params=generated_beta_params,
                                                                        learning_rate_penalty=learning_rate,
                                                                        num_step=current_step_idx)
            name_weight_momentum = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                        names_grads_wrt_params_dict=momentum,
                                                                        generated_alpha_params=generated_alpha_params,
                                                                        generated_beta_params=generated_beta_params,
                                                                        learning_rate_penalty=learning_rate,
                                                                        num_step=current_step_idx)
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1   
            names_weights_grad = {
               name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
                 name, value in names_weights_grad.items()}  
            name_weight_momentum = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
                 name, value in name_weight_momentum.items()}                         
            support_loss_grad, support_preds_grad, support_loss_s= self.net_forward(x=support_data,
                                                        y=support_label,
                                                        weights=names_weights_grad,
                                                        backup_running_statistics=False,
                                                        training=True, num_step=num_step)

            support_loss_momentum, support_preds_grad,support_loss_s = self.net_forward(x=support_data,
                                                        y=support_label,
                                                        weights=name_weight_momentum,
                                                        backup_running_statistics=False,
                                                        training=True, num_step=num_step)
            '''
            for k in names_weights_copy:
              if random.uniform(0, 1) < 0.3:
                names_grad_negative[k] = -1*names_grads_copy[k]
                names_momentum_negative[k] = -1*momentum[k]
              else:
                names_grad_negative[k] = names_grads_copy[k]
                names_momentum_negative[k] = momentum[k]
            names_weights_grad_negative = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                        names_grads_wrt_params_dict=names_grad_negative,
                                                                        generated_alpha_params={},
                                                                        generated_beta_params={},
                                                                        num_step=current_step_idx)
            names_weights_momentum_negative = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                        names_grads_wrt_params_dict=names_momentum_negative,
                                                                        generated_alpha_params={},
                                                                        generated_beta_params={},
                                                                        num_step=current_step_idx)
            names_weights_grad_negative = {
               name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
                 name, value in names_weights_grad_negative.items()}  
            names_weights_momentum_negative = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
                 name, value in names_weights_momentum_negative.items()}          
            support_loss_grad_negative, support_preds_grad = self.net_forward(x=support_data,
                                                        y=support_label,
                                                        weights=names_weights_grad_negative,
                                                        backup_running_statistics=False,
                                                        training=True, num_step=num_step)

            support_loss_momentum_negative, support_preds_grad = self.net_forward(x=support_data,
                                                        y=support_label,
                                                        weights=names_weights_momentum_negative,
                                                        backup_running_statistics=False,
                                                        training=True, num_step=num_step)
            
          
            '''
            loss_total = torch.stack((torch.sub(support_loss_grad,loss),torch.sub(support_loss_momentum,loss)))
            loss_total = -loss_total
            weight = F.softmax(loss_total,dim = 0)
            weight_grad = weight[0].item()
            weight_momentum = weight[1].item()
            if support_loss_momentum > loss:
               loss_count  = loss_count + 1
            for k in learning_rate:
                if learning_rate[k] < 0.3:
                    learning_rate[k] = 1
                if support_loss_momentum < loss:    
                    names_grads_copy[k] = weight_grad*names_grads_copy[k] + weight_momentum*momentum[k]
                        #cnames_grads_copy[k] = names_grads_copy[k]/(1-weight_momentum**num_step)
        momentum = names_grads_copy   
        #names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}
        #for key, grad in names_grads_copy.items():
            #if grad is None:
                #print('Grads not found for inner loop parameter', key)
            #names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     generated_alpha_params=generated_alpha_params,
                                                                     generated_beta_params=generated_beta_params,
                                                                     learning_rate_penalty={},
                                                                     num_step=current_step_idx)
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}


        return names_weights_copy,momentum,cos_sim,loss_count

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase,loss_count):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.net_forward
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        total_support_accuracies = [[] for i in range(num_steps)]
        total_target_accuracies = [[] for i in range(num_steps)]
        per_task_target_preds = [[] for i in range(len(x_target_set))]

        if torch.cuda.device_count() > 1:
            self.classifier.module.zero_grad()
        else:
            self.classifier.zero_grad()
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []
            task_accuracies = []
            per_step_support_accuracy = []
            per_step_target_accuracy = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
            
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}
            

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            # Attenuate the initialization for L2F
            if self.args.attenuate:
                # Obtain gradients from support set for task embedding
                task_embeddings = self.get_task_embeddings(x_support_set_task=x_support_set_task,
                                                           y_support_set_task=y_support_set_task,
                                                           names_weights_copy=names_weights_copy)
                
                names_weights_copy = self.attenuate_init(task_embeddings=task_embeddings,
                                                                                         names_weights_copy=names_weights_copy)                                          
            momentum = {}    
            for k, v in names_weights_copy.items():
                momentum[k] = torch.zeros_like(v)                                        
            learning_rate = {}
            cos_sim_total = []
            device = torch.device('cuda') 
            h0 = torch.zeros(2,1,int(len(names_weights_copy))).to(device)
            c0 = torch.zeros(2,1,int(len(names_weights_copy))).to(device)
            h1 = torch.zeros(2,1,int(len(names_weights_copy))).to(device)
            c1 = torch.zeros(2,1,int(len(names_weights_copy))).to(device)
            for num_step in range(num_steps):
                support_loss, support_preds, supp_loss_seperate = self.net_forward(x=x_support_set_task,
                                                               y=y_support_set_task, 
                                                               weights=names_weights_copy,
                                                               backup_running_statistics=
                                                               True if (num_step == 0) else False,
                                                               training=True, num_step=num_step)
                support_loss_grad = torch.autograd.grad(support_loss, names_weights_copy.values(), retain_graph=True)
 
                generated_alpha_params = {}
                generated_beta_params = {}
                if self.args.alfa:
                    '''
                    names_grads_copy = dict(zip(names_weights_copy.keys(), support_loss_grad))                                                     
                    names_weights_copy_l = {key: value[0] for key, value in names_weights_copy.items()}
                    for key, grad in names_grads_copy.items():
                            if grad is None:
                                print('Grads not found for inner loop parameter', key)
                            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                    if num_step > 0:

                        names_weights_grad = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy_l,
                                                                            names_grads_wrt_params_dict=names_grads_copy,
                                                                            generated_alpha_params={},
                                                                            generated_beta_params={},
                                                                            learning_rate_penalty={},
                                                                            num_step=num_step)
                        name_weight_momentum = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy_l,
                                                                            names_grads_wrt_params_dict=momentum,
                                                                            generated_alpha_params={},
                                                                            generated_beta_params={},
                                                                            learning_rate_penalty={},
                                                                            num_step=num_step)                                     
                        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1 
                        names_weights_grad = {
                                name.replace('module.', ''): value.unsqueeze(0).repeat(
                                    [num_devices] + [1 for i in range(len(value.shape))]) for
                                    name, value in names_weights_grad.items()}  
                        name_weight_momentum = {
                                name.replace('module.', ''): value.unsqueeze(0).repeat(
                                [num_devices] + [1 for i in range(len(value.shape))]) for
                                name, value in name_weight_momentum.items()}                         
                        support_loss_grad_ahead, support_preds_grad, support_loss_s_g= self.net_forward(x=x_support_set_task,
                                                        y=y_support_set_task,
                                                        weights=names_weights_grad,
                                                        backup_running_statistics=False,
                                                        training=True, num_step=num_step)
                        support_loss_momentum_ahead, support_preds_grad,support_loss_s_m = self.net_forward(x=x_support_set_task,
                                                        y=y_support_set_task,
                                                        weights=name_weight_momentum,
                                                        backup_running_statistics=False,
                                                        training=True, num_step=num_step)
                        supp_l_list = torch.split(torch.sub(support_loss_s_g, support_loss_s_m),1)
                    else:
                        names_weights_grad = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy_l,
                                                                            names_grads_wrt_params_dict=names_grads_copy,
                                                                            generated_alpha_params={},
                                                                            generated_beta_params={},
                                                                            learning_rate_penalty={},
                                                                            num_step=num_step)
                    
                        names_weights_grad = {
                                name.replace('module.', ''): value.unsqueeze(0).repeat(
                                    [num_devices] + [1 for i in range(len(value.shape))]) for
                                    name, value in names_weights_grad.items()}                  
                        support_loss_grad_ahead, support_preds_grad, support_loss_s_g= self.net_forward(x=x_support_set_task,
                                                        y=y_support_set_task,
                                                        weights=names_weights_grad,
                                                        backup_running_statistics=False,
                                                        training=True, num_step=num_step)
                        supp_l_list = torch.split(support_loss_s_g, 1)
                        for i in range(len(supp_l_list)):
                            supp_l_list[i][0] = 0
                    '''
                    per_step_task_embedding_lr = []
                    per_step_task_embedding_wd = []
                    '''
                    for k, v in names_weights_copy.items():
                        per_step_task_embedding.append(v.mean())
                    '''

                    for i in range(len(support_loss_grad)):
                        per_step_task_embedding_lr.append(support_loss_grad[i].mean())

                    if num_step > 0:
                        for k, v in momentum.items():
                            per_step_task_embedding_lr.append(v.mean())
                    else:
                        for i in range(len(support_loss_grad)):
                            per_step_task_embedding_lr.append(support_loss_grad[i].mean())
                    
                    for k, v in names_weights_copy.items():
                        per_step_task_embedding_wd.append(v.mean())

                    for i in range(len(support_loss_grad)):
                        per_step_task_embedding_wd.append(support_loss_grad[i].mean())  
                    '''
                    for l in supp_l_list:
                        per_step_task_embedding.append(l[0])
                    '''
                    per_step_task_embedding_lr = torch.stack(per_step_task_embedding_lr)
                    per_step_task_embedding_lr = torch.reshape(per_step_task_embedding_lr,(1,1,int(len(support_loss_grad))*2))
                    per_step_task_embedding_wd = torch.stack(per_step_task_embedding_wd)
                    per_step_task_embedding_wd = torch.reshape(per_step_task_embedding_wd,(1,1,int(len(support_loss_grad))*2))

                    generated_params_alpha, (h0,c0) = self.regularizer1(per_step_task_embedding_lr, (h0,c0))
                    generated_params_beta, (h1,c1) = self.regularizer2(per_step_task_embedding_wd, (h1,c1))
                    num_layers = len(names_weights_copy)
                    #generated_alpha, generated_beta = torch.split(generated_params_alpha[0][0], split_size_or_sections=num_layers)
                    generated_alpha = torch.reshape(generated_params_alpha,(num_layers,1))
                    generated_beta = torch.reshape(generated_params_beta,(num_layers,1))

                    #generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=num_layers)
                    g = 0
                    for key in names_weights_copy.keys():
                        generated_alpha_params[key] =generated_alpha[g]*2
                        generated_beta_params[key] = generated_beta[g]
                        g+=1
                start = time.time()
                names_weights_copy, momentum,cos_sim,loss_count = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  generated_beta_params=generated_beta_params,
                                                                  generated_alpha_params=generated_alpha_params,
                                                                  use_second_order=use_second_order,
                                                                  momentum = momentum,
                                                                  current_step_idx=num_step,
                                                                  support_data = x_support_set_task,
                                                                  support_label= y_support_set_task,
                                                                  num_step = num_step,
                                                                  learning_rate=learning_rate,
                                                                  loss_count=loss_count)
                end = time.time()
                alpha = []
                for key in generated_alpha_params:
                    alpha.append(generated_alpha_params[key].item())
                if alpha != []:
                   cos_sim_total.append(alpha)
                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds,target_l_s = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)
                    
                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)

                else:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):
                        target_loss, target_preds, target_l_s = self.net_forward(x=x_target_set_task,
                                                                     y=y_target_set_task, weights=names_weights_copy,
                                                                     backup_running_statistics=False, training=True,
                                                                     num_step=num_step)
                        task_losses.append(target_loss)
             
            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                if torch.cuda.device_count() > 1:
                    self.classifier.module.restore_backup_stats()
                else:
                    self.classifier.restore_backup_stats()
        
        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)
        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()
        return losses, per_task_target_preds,generated_alpha_params,generated_beta_params,learning_rate,cos_sim_total,loss_count

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step)
        loss = F.cross_entropy(input=preds, target=y)
        loss_seperate = F.cross_entropy(input=preds,target=y,reduction='none')
        if 0:
            supp_loss_class = loss_seperate.reshape(5,-1)
            supp_loss_avg = torch.mean(supp_loss_class,dim=1)
            loss_weight = torch.softmax(supp_loss_avg/20,dim=0).detach()
            loss = F.cross_entropy(input=preds, target=y, weight=loss_weight)
        return loss, preds, loss_seperate

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch, loss_count):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds,current_alpha,current_beta,learning_rate,cos_sim,loss_count = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True,
                                                     loss_count= loss_count)
        return losses, per_task_target_preds,current_alpha,current_beta,learning_rate,cos_sim, loss_count

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds,current_alpha,current_beta,lr,cs,loss_c = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False,
                                                     loss_count = 0)

        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        start = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        end = time.time()
        print(end-start)
        #if 'imagenet' in self.args.dataset_name:
        #    for name, param in self.classifier.named_parameters():
        #        if param.requires_grad:
        #            param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        #for name, param in self.classifier.named_parameters():
        #    print(param.mean())

        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch,loss_count):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds,alpha,beta,lr,cos_sim, loss_count = self.train_forward_prop(data_batch=data_batch, epoch=epoch, loss_count = loss_count)
        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds,alpha,beta,lr,cos_sim, loss_count

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        #losses['loss'].backward() # uncomment if you get the weird memory error
        self.zero_grad()
        self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
