from interpensembles.layers import Interp_Conv2d_factory
import torch
import numpy as np


class Test_Interp_Conv2d_factory():
    def test_init(self):
        "Test that initialization works properly."
        nb_subnets = 4
        in_channels = 64
        out_channels = 64
        kernel_size = 3
        convfactory = Interp_Conv2d_factory(nb_subnets,in_channels,out_channels,kernel_size)

    def test_create_subnet_params(self):    
        """Test the created subnets have the right parameters: make sure zeros are in the right place.
        """
        nb_subnets = 4
        in_channels = 64
        out_channels = 64
        kernel_size = 3
        convfactory = Interp_Conv2d_factory(nb_subnets,in_channels,out_channels,kernel_size)
        for i in range(nb_subnets):
            x_ind = int(i%np.sqrt(nb_subnets))
            y_ind = int(i//np.sqrt(nb_subnets))
            x_thresh_low = int(x_ind*in_channels/np.sqrt(nb_subnets))
            y_thresh_low = int(y_ind*in_channels/np.sqrt(nb_subnets))
            x_thresh_high = int((x_ind+1)*in_channels/np.sqrt(nb_subnets))
            y_thresh_high = int((y_ind+1)*in_channels/np.sqrt(nb_subnets))
            ## the subblock that matches the main weights should be identical to the base weights. 
            assert torch.all(convfactory.subnets[i].get_masked()[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high,]== convfactory.base_weights[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]) 
            ## all other subblocks of weights should be 0
            others = list(range(nb_subnets))
            others.pop(i)        
            for o in others:
                assert torch.all(convfactory.subnets[o].get_masked()[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]== 0) 
            assert torch.all(convfactory.subnets[0].bias != convfactory.base_bias)    

    def test_gradients_both(self):    
        """Test that gradient flow works correctly- flows through the weights from all subnets and the base network, and also does not change the masks.. 

        """
        nb_subnets = 4
        in_channels = 64
        out_channels = 64
        kernel_size = 3

        convfactory = Interp_Conv2d_factory(nb_subnets,in_channels,out_channels,kernel_size).double()
        orig_masked = convfactory.subnets[0].get_masked()

        ## now we calculate a loss off of a base dummy input. 
        dummy = torch.tensor(np.ones((1,64,64,64)))
        dummy_out = convfactory.base_convnet(dummy)+convfactory.subnets[0](dummy)+ convfactory.subnets[1](dummy)+ convfactory.subnets[2](dummy)+ convfactory.subnets[3](dummy) ## work off of just the first subnet added to the base convnet
        #dummy_avg = [convfactory.subnets[i](dummy) for i in range(nb_subnets)]

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(convfactory.parameters(),lr = 0.1,momentum=0.9)
        optimizer.zero_grad() 
        loss_fn(dummy_out,dummy[:,:,:62,:62]).backward()
        optimizer.step()
        optimizer.step()
        for i in range(nb_subnets):
            x_ind = int(i%np.sqrt(nb_subnets))
            y_ind = int(i//np.sqrt(nb_subnets))
            x_thresh_low = int(x_ind*in_channels/np.sqrt(nb_subnets))
            y_thresh_low = int(y_ind*in_channels/np.sqrt(nb_subnets))
            x_thresh_high = int((x_ind+1)*in_channels/np.sqrt(nb_subnets))
            y_thresh_high = int((y_ind+1)*in_channels/np.sqrt(nb_subnets))
            ## the subblock that matches the main weights should be identical to the base weights. 
            masked = convfactory.subnets[i].get_masked()
            assert torch.all(masked[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high,]== convfactory.base_convnet.weight[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]) 
            # all other subblocks of weights should be 0
            others = list(range(nb_subnets))
            others.pop(i)        
            for o in others:
                masked = convfactory.subnets[o].get_masked()
                assert torch.all(masked[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]== 0) 
            ## The weights should be different from the original masked weights
            assert torch.any(masked!=orig_masked) 
            ## The biases should be different from the base network's biases. 
            assert torch.any(convfactory.subnets[i].bias != convfactory.base_convnet.bias)
        
    def test_gradients_base(self):    
        """Test that gradient flow works correctly- flows through the weights from the base network, but still changes weights in the sublayers, and also does not change the masks.. 

        """
        nb_subnets = 4
        in_channels = 64
        out_channels = 64
        kernel_size = 3

        convfactory = Interp_Conv2d_factory(nb_subnets,in_channels,out_channels,kernel_size).double()
        orig_masked = convfactory.subnets[0].get_masked()

        ## now we calculate a loss off of a base dummy input. 
        dummy = torch.tensor(np.ones((1,64,64,64)))
        dummy_out = convfactory.base_convnet(dummy) ## work off of just the output from base convnet
        #dummy_avg = [convfactory.subnets[i](dummy) for i in range(nb_subnets)]

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(convfactory.parameters(),lr = 0.1,momentum=0.9)
        optimizer.zero_grad() 
        loss_fn(dummy_out,dummy[:,:,:62,:62]).backward()
        optimizer.step()
        optimizer.step()
        for i in range(nb_subnets):
            x_ind = int(i%np.sqrt(nb_subnets))
            y_ind = int(i//np.sqrt(nb_subnets))
            x_thresh_low = int(x_ind*in_channels/np.sqrt(nb_subnets))
            y_thresh_low = int(y_ind*in_channels/np.sqrt(nb_subnets))
            x_thresh_high = int((x_ind+1)*in_channels/np.sqrt(nb_subnets))
            y_thresh_high = int((y_ind+1)*in_channels/np.sqrt(nb_subnets))
            ## the subblock that matches the main weights should be identical to the base weights. 
            masked = convfactory.subnets[i].get_masked()
            assert torch.all(masked[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high,]== convfactory.base_convnet.weight[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]) 
            # all other subblocks of weights should be 0
            others = list(range(nb_subnets))
            others.pop(i)        
            for o in others:
                masked = convfactory.subnets[o].get_masked()
                assert torch.all(masked[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]== 0) 
            ## The weights should be different from the original masked weights
            assert torch.any(masked!=orig_masked) 
            ## The biases should be different from the base network's biases. 
            assert torch.any(convfactory.subnets[i].bias != convfactory.base_convnet.bias)

    def test_gradients_sub(self):    
        """Test that gradient flow works correctly- flows through the weights from the base network, but still changes weights in the sublayers, and also does not change the masks.. 

        """
        nb_subnets = 4
        in_channels = 64
        out_channels = 64
        kernel_size = 3

        convfactory = Interp_Conv2d_factory(nb_subnets,in_channels,out_channels,kernel_size).double()
        orig_masked = convfactory.subnets[0].get_masked()

        ## now we calculate a loss off of a base dummy input. 
        dummy = torch.tensor(np.ones((1,64,64,64)))
        dummy_out = convfactory.subnets[0](dummy)+ convfactory.subnets[1](dummy)+ convfactory.subnets[2](dummy)+ convfactory.subnets[3](dummy) ## work off of just the first subnet added to the base convnet
        #dummy_avg = [convfactory.subnets[i](dummy) for i in range(nb_subnets)]

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(convfactory.parameters(),lr = 0.1,momentum=0.9)
        optimizer.zero_grad() 
        loss_fn(dummy_out,dummy[:,:,:62,:62]).backward()
        optimizer.step()
        optimizer.step()
        for i in range(nb_subnets):
            x_ind = int(i%np.sqrt(nb_subnets))
            y_ind = int(i//np.sqrt(nb_subnets))
            x_thresh_low = int(x_ind*in_channels/np.sqrt(nb_subnets))
            y_thresh_low = int(y_ind*in_channels/np.sqrt(nb_subnets))
            x_thresh_high = int((x_ind+1)*in_channels/np.sqrt(nb_subnets))
            y_thresh_high = int((y_ind+1)*in_channels/np.sqrt(nb_subnets))
            ## the subblock that matches the main weights should be identical to the base weights. 
            masked = convfactory.subnets[i].get_masked()
            assert torch.all(masked[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high,]== convfactory.base_convnet.weight[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]) 
            # all other subblocks of weights should be 0
            others = list(range(nb_subnets))
            others.pop(i)        
            for o in others:
                masked = convfactory.subnets[o].get_masked()
                assert torch.all(masked[y_thresh_low:y_thresh_high,x_thresh_low:x_thresh_high]== 0) 
            ## The weights should be different from the original masked weights
            assert torch.any(masked!=orig_masked) 
            ## The biases should be different from the base network's biases. 
            assert torch.any(convfactory.subnets[i].bias != convfactory.base_convnet.bias)




