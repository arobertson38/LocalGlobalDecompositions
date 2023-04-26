"""
This file allows the error suite to work correctly. 
I swear its the most finky code because of all the
relative imports that it has to do. 

"""
import MKS.utils.TestingSuite as ts
from MKS.utils.HelperFunctions import loss_SBG
import argparse
import torch
import os
import numpy as np
import MKS.utils.HelperFunctions as hfunc

def binarize(structs, means=None):
    """
    This method discretizes a microstructure (maps it to
    0 and 1) while enforcing the mean.

    This is done following a congruent method to what is 
    used in the post-processing from the previous paper. 

    ############################################
    STRUCTURE MUST BE THE FORM (#xSPATIAL DIMENSIONS)
        - can be 2D or 3D
    ############################################
    """
    structs = structs.detach().numpy()

    if means is None:
        means = np.mean(structs, axis=tuple(range(1, len(structs.shape))))
    elif type(means) is float:
        means = np.ones((len(structs),)) * means

    N = np.product(structs.shape[1:])

    new_structures = np.zeros_like(structs)

    for n, (struct, mean) in enumerate(zip(structs, means)):
        index = np.unravel_index(np.argpartition(struct, -int(N * mean),
                             axis=None)[-int(N * mean):], struct.shape)

        new_structures[n][index] = 1.0

        #print(f"SBG Mean: {struct.mean()}. Desired Mean: {mean}.")
        #f, ax = plt.subplots(1, 3, figsize=[12, 5])
        #ax[0].imshow(struct, cmap='gray')
        #ax[1].imshow(new_structures[n], cmap='gray')
        #ax[2].imshow(np.abs(struct - new_structures[n]), cmap='gray')
        #plt.show()

    return torch.from_numpy(new_structures).float()


if __name__ == "__main__":
    # Testing Suite for error
    parser = argparse.ArgumentParser(description='Automatically test output of SBG/SDE models.')
    parser.add_argument("--image_folder", required=True, help="The save directory")
    parser.add_argument("--micro",
        type=str, 
        default='uncond_samples.pth',
        help="File name for the microstructures we are testing.")
    parser.add_argument("--mode",
            type=str,
            default='exp',
            help="The data to compare against (NBSA, TI)."
            )
    parser.add_argument("--stdout",
            required=True,
            type=str,
            help="The location of the STDOUT file"
            )

    
    args = parser.parse_args()
    data = torch.load(os.path.join(args.image_folder, args.micro))

    # test:
    if args.mode.lower() == "nbsa":
        # its the Case Study 1 data and should be tested as such. 
        tester = ts.TestingSuite(mode=args.mode)
        data = binarize(data)
    elif args.mode.lower() == 'ti':
        tester = ts.TestingSuite_3Phase(mode='TI64')
        data = hfunc.threshold_torch(data).float()
    tester.complete_suite(data, args.image_folder)

    # saving the image of the training error as well
    loss_SBG(os.path.join(args.stdout, 'stdout.txt'), \
            os.path.join(args.image_folder, 'TestTrainLoss.png'))

    # displaying some examples as well
    ts.save_plot_examples(data, \
            os.path.join(args.image_folder, 'Comparison.png'))

