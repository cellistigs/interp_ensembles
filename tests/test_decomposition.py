## tests for script plot_conditional_variance.py. 

## test that Var, JS divergence behave as expected. 
import os 
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here,"../scripts/"))
from plot_conditional_variance import Var,JS
import numpy as np
import torch
nclasses = 10
nmodels = 10 
ndata = 20
labels = torch.tensor(np.array([1 for i in range(ndata)]))

class Test_JS():
    """test that multi-class Jensen-Shannon Divergence works as expected. 

    JS expects probs, shaped as (sample,model,class), and labels shaped as (sample?)
    """

    def test_min(self):
        """Test that jensen shannon is minimized when all identical and perfectly certain. 

        """
        numpy_perfect_certain_agree = np.tile(np.array([[int(j == 0)+1e-12*(j!=0) for j in range(nclasses)] for i in range((nmodels))])[None,:,:],(ndata,1,1))
        perfect_certain_agree = torch.tensor(numpy_perfect_certain_agree).double()
        assert np.allclose(JS(perfect_certain_agree,labels)[0].numpy(),0) 
        assert np.allclose(JS(perfect_certain_agree,labels)[1].numpy(),0) 
        

    def test_min_h(self):    
        """Test that jensen shannon is minimized when all identical and perfectly uncertain. 

        """
        numpy_perfect_uncertain= np.tile(np.array([1/nclasses for i in range(nclasses)])[None,None,:],(ndata,nmodels,1)) 
        perfect_uncertain = torch.tensor(numpy_perfect_uncertain).double()
        assert np.allclose(JS(perfect_uncertain,labels)[0].numpy(), np.log(nclasses))
        assert np.all(JS(perfect_uncertain,labels)[1].numpy()) == 0

    def test_max_h(self):    
        """Test that jensen shannon is maximized when all differ and perfectly certain. 

        """
        numpy_perfect_certain_disagree = np.tile(np.array([[int(j == i)+1e-12*(j!=i) for j in range(nclasses)] for i in range((nmodels))])[None,:,:],(ndata,1,1))
        perfect_certain_disagree = torch.tensor(numpy_perfect_certain_disagree).double()
        assert np.allclose(JS(perfect_certain_disagree,labels)[0].numpy(),0) 
        assert np.allclose(JS(perfect_certain_disagree,labels)[1].numpy(),np.log(nclasses)) 



