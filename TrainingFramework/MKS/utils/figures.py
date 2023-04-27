"""
A file containing lots of functions for making figures. 
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as tf
#from GeneratingDataset import flood
from matplotlib.gridspec import GridSpec
from . import HelperFunctions as helpers
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import MMD_functions as mmd
import pickle

def show(trans, pc1, pc2, labels=None, distinct_labels=None, \
        ax=None, return_ax=False, show=True, ms=4, \
        labelels=None):
    # Parameters
    colors = ['k.', 'r.', 'c.', 'b.', 'm.', 'g.']
    
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)

    ax.ticklabel_format(style='sci', scilimits=(-2,2))

    labelels = [f'Class {n+1}' for n in range(len(distinct_labels))] \
            if (labelels is None) else labelels

    if labels is None:
        ax.plot(trans[:, pc1], trans[:, pc2], 'k.')
        ax.set_xlabel(f"PC {pc1 + 1}")
        ax.set_ylabel(f"PC {pc2 + 1}")
    else:
        ax.set_xlabel(f"PC {pc1 + 1}")
        ax.set_ylabel(f"PC {pc2 + 1}")
        for n, lab in enumerate(distinct_labels):
            ax.plot(trans[labels==lab, pc1], \
                    trans[labels==lab, pc2], colors[lab], \
                    ms=ms, label=labelels[lab])
        ax.legend()

    if show:
        plt.show()
    elif return_ax:
        return ax
    else:
        pass

def show_scree(scree, ax=None, return_ax=False, show=True):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)

    ax.plot(np.arange(len(scree)), np.cumsum(scree), 'kx-')
    ax.set_xlabel('Cumulative Components')
    ax.set_ylabel('Percent Variance Explained')

    if show:
        plt.show()
    elif return_ax:
        return ax
    else:
        pass

def thresh(image, cut=0.5):
    image_copy = image.clone()
    image_copy[image_copy >= cut] = 1.0
    image_copy[image_copy < 1.0] = 0.0
    return image_copy

def plot(im, ax, letter=None, thresh=thresh):
    ax.imshow(thresh(im.squeeze()), cmap='gray')
    ax.text(-0.15, 1.03, letter, transform=ax.transAxes, size=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

# -------------------------------------------------
# Round 2 Figures
# -------------------------------------------------

def plot_PCA_R2():
    """
    Plot the PCA results from round 2 of this process. 
    """
    folder = 'expsyn_40x40_PCA'
    trans = np.load(f'./GenerationResults/{folder}/trans.npy')
    labels = torch.load(f'./GenerationResults/{folder}/labels.pth').numpy()

    stats_sbg = torch.load('./GenerationResults/N40_SYN_V1_RI2PS.pth')
    with open(f'./GenerationResults/{folder}/pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    trans_sbg = pca.transform(stats_sbg.view(stats_sbg.shape[0], -1).numpy())
    del stats_sbg

    trans = np.concatenate([trans, trans_sbg], axis=0)
    labels = np.concatenate([labels, 2 * np.ones(trans_sbg.shape[0])], axis=0)


    f = plt.figure(figsize=[8, 4])

    distinct = [[0, 1, 2], [0, 1], [0, 2]]

    pc1 = 8
    pc2 = 9

    for i in range(3):
        ax = f.add_subplot(1, 3, 1+i)
        show(trans, \
             pc1, \
             pc2, \
             labels, \
             distinct[i], \
             ms = 2.2, \
             ax=ax, \
             show=False, \
             labelels = ['Real', 'Syn (GRF)', 'Syn (SBG)'])

    # ------------------------------------------
    # Computing the MMD Distance between points in the PCA projection
    # ------------------------------------------
    mmd_flag = True
    if mmd_flag:
        subset = 1000
        class0 = torch.from_numpy(trans[labels==0, :])
        class0 = class0[torch.randperm(len(class0))[:1000], :]

        class1 = torch.from_numpy(trans[labels==1, :])
        class1 = class1[torch.randperm(len(class1))[:1000], :]

        class2 = torch.from_numpy(trans[labels==2, :])
        class2 = class2[torch.randperm(len(class2))[:1000], :]

        classes = [class0, class1, class2]
        sigma_list = [1, 5, 10, 15]

        for i in range(3):
            for j in range(i+1, 3):
                print(f"MMD Distance Between Class {i+1} and Class {j+1} is: {mmd.mix_rbf_mmd2(classes[i], classes[j], sigma_list)}.")

    f.tight_layout()
    plt.show()








# -------------------------------------------------
# Original Report Figures
# -------------------------------------------------

def comp_sbg_to_closest():
    """
    This method creates a figure comparing patches. 
    It compares a synthetically generated patch
    to a refined version of it to the closest
    patch in the training dataset. 
    """

    data = torch.load('./GenerationResults/conditional_small_original.pth').squeeze()
    data_new = torch.load('./GenerationResults/conditional_small_SigmaIndex2.pth').squeeze()
    training = torch.load('./data/76XX_V3_40x40_CUTDOWN.pth').squeeze()

    images = [10, 30, 26, 42]

    f = plt.figure()

    for n, image_index in enumerate(images):

        diff = (training - data[image_index, ...]).view(training.shape[0], -1)
        diff = torch.linalg.norm(diff, dim=-1)
        index = torch.argmin(diff)

        closest = training[index, ...]
        

        ax = f.add_subplot(len(images), 3, 3*n + 1)
        ax.imshow(thresh(data[image_index, ...]), cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if n == 0: ax.set_title('Synthetic Patch')

        ax = f.add_subplot(len(images), 3, 3*n + 2)
        ax.imshow(thresh(data_new[image_index, ...]), cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if n == 0: ax.set_title('Refined Patch')

        ax = f.add_subplot(len(images), 3, 3*n + 3)
        ax.imshow(thresh(closest), cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        if n == 0: ax.set_title('Closest Patch')

    f.tight_layout()
    plt.show()


def select_patch_size():
    """
    This method creates a figure comparing various patch sizes. 
    """
    data = torch.load('./data/76XX_Synthetic_V5_LargeImage_StructIndex2BASE_RESIZE.pth')
    data = data.squeeze()[0, ...]
    sizes = [30, 40, 50]

    f = plt.figure()

    for n, si in enumerate(sizes):
        crop = tf.RandomCrop(si)
        ax = f.add_subplot(1, len(sizes), n+1)
        ax.imshow(thresh(crop(data.unsqueeze(0)).squeeze()), cmap='gray')
        ax.set_title(f"Size: {si}x{si}")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

#def demonstrate_dataset_generation():
#    """
#    This method creates a figure demonstrating the generation
#    of the dataset. 
#    """
#    data = torch.load('./data/olddata/76XX_NBSA_CUTDOWN.pth')[..., 14].unsqueeze(0)
#
#    crop = tf.RandomCrop(111)
#    rot = tf.RandomRotation([0, 360], fill=(2,))
#
#    cropped = crop(data)
#    
#    flag = False
#    while not flag:
#        roted = crop(rot(data))
#        if (roted != 2).all():
#            flag = True
#
#    emptied = flood(cropped.squeeze(), (0, 0), 0.4)
#
#    # resizing
#    resize = tf.Resize(40)
#    cropped = resize(cropped)
#    roted = resize(roted)
#    emptied = resize(emptied.unsqueeze(0))
#    empty = torch.zeros([1, 40, 40])
#
#    f = plt.figure()
#    gs1 = GridSpec(1, 2)
#    gs2 = GridSpec(2, 4)
#
#    ax = f.add_subplot(gs1[0])
#    plot(data, ax, '(a)')
#    
#    ax = f.add_subplot(gs2[2])
#    plot(cropped, ax, '(b)')
#
#    ax = f.add_subplot(gs2[3])
#    plot(roted, ax, '(c)')
#
#    ax = f.add_subplot(gs2[6])
#    plot(emptied, ax, '(d)')
#
#    ax = f.add_subplot(gs2[7])
#    plot(empty, ax, '(e)')
#
#    plt.show()


def demonstrate_closest_patch():
    data = torch.load('./ncsn/images/image_raw_999.pth')
    training = torch.load('./data/76XX_V3_40x40_CUTDOWN.pth').squeeze()

    images = [5, 6, 20, 2]

    f = plt.figure()

    for n, image_index in enumerate(images):

        diff = (training - data[image_index, ...]).view(training.shape[0], -1)
        diff = torch.linalg.norm(diff, dim=-1)
        index = torch.argmin(diff)

        closest = training[index, ...]
        

        ax = f.add_subplot(len(images), 2, 2*n + 1)
        plot(data[image_index, ...], ax)
        if n == 0: ax.set_title('Generated Patch')


        ax = f.add_subplot(len(images), 2, 2*n + 2)
        plot(closest, ax)
        if n == 0: ax.set_title('Closest Patch')

    dist = []
    for image_index in range(data.shape[0]):
        diff = (training - data[image_index, ...]).view(training.shape[0], -1)
        diff = torch.linalg.norm(diff, dim=-1)
        index = torch.argmin(diff)

        closest = training[index, ...]
        dist.append(torch.linalg.norm(data[image_index, ...] - closest) /\
                torch.linalg.norm(data[image_index, ...]))

    print(f"Average Relative Distance: {torch.tensor(dist).mean()}.")


    f.tight_layout()
    plt.show()

def demonstrate_refinement_sigmas():
    """
    This method demonstrates the refinement induced by
    2 different sigmas. 
    """
    data = torch.load('./GenerationResults/conditional_small_original.pth').squeeze()
    data_new = torch.load('./GenerationResults/conditional_small_SigmaIndex2.pth').squeeze()
    data_new_s1 = torch.load('./GenerationResults/conditional_small_SigmaIndex1.pth').squeeze()
    index = 29

    f = plt.figure()
    ax = f.add_subplot(131)
    plot(data[index, ...], ax).set_title('Synthetic Patch')

    ax = f.add_subplot(132)
    plot(data_new[index, ...], ax).set_title('Sigma: 0.45')

    ax = f.add_subplot(133)
    plot(data_new_s1[index, ...], ax).set_title('Sigma: 0.67')

    f.tight_layout()
    plt.show()

def twentyfiverefined():
    """
    This method generates 25 refined images
    """
    data_new = torch.load('./GenerationResults/conditional_small_SigmaIndex2.pth').squeeze()
    
    f = plt.figure()
    for i in range(25):
        ax = f.add_subplot(5, 5, i+1)
        plot(data_new[i, ...], ax)

    f.tight_layout()
    plt.show()

def large_refinement():
    """
    This method generates a figure displaying the
    results of the large refinement. 
    """
    data = torch.load('./GenerationResults/conditional_generation_large_SigmaIndex2_StructIndex3.pth')
    images = [0, 1, 2, 3]

    f = plt.figure()
    for n, image_index in enumerate(images):
        ax = f.add_subplot(2, 2, n+1)
        plot(data[image_index, ...], ax)

    f.tight_layout()
    plt.show()

def compare_statistics():
    """
    This method compares the statistics for a local patch
    """
    data = torch.load('./GenerationResults/conditional_small_SigmaIndex2.pth').squeeze()[10, ...].numpy()
    stats = helpers.twopoint_np(data, data, 1.0, centered=True)
    stats_rot, rho, theta = helpers.rotational_invariance_2D(stats, \
            centered=True,
            crop_rho=0.3,
            return_rho_theta=True)
    stats = helpers.twopoint_np(data, data, 0.5, centered=True)


    f = plt.figure()
    ax = f.add_subplot(131)
    ax.imshow(thresh(torch.from_numpy(data)), cmap='gray')
    ax.set_title('Patch')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = f.add_subplot(133)
    im = ax.imshow(stats)
    ax.set_xticks([0, 9, 18])
    ax.set_yticks([0, 9, 18])
    ax.set_xticklabels(['-9', '0', '9'])
    ax.set_yticklabels(['-9', '0', '9'])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    f.colorbar(im, ax=ax, cax=cax)


    ax = f.add_subplot(132, projection='polar')
    ax.contourf(theta, rho, stats_rot)

    f.tight_layout()
    plt.show()

def plot_PCA():
    """
    This method plots the results of PCA. 
    """
    folder = 'PCAAnalysis/AhmetResults_Filtered'
    trans = np.load('./' + folder + '/trans.npy')
    labels = np.load('./' + folder + '/labels.npy')
    scree = np.load('./' + folder + '/cumev.npy')
    ms = 0.6
    
    f = plt.figure()
    ax = f.add_subplot(121)
    show(trans, 3, 7, labels, [0, 1], ms=ms, ax=ax, show=False)

    ax = f.add_subplot(122)
    show_scree(scree, ax=ax, show=False)
    
    f.tight_layout()
    plt.show()
    
def plot_J2():
    """
    plots the J2 clustering progression
    """
    import pickle
    with open('./GenerationResults/J2Results.pkl', 'rb') as f:
        data = torch.tensor(pickle.load(f)[1])

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(torch.arange(0, len(data)), data, 'kx')
    ax.set_xlabel("J2 Value for Clusters")
    ax.set_ylabel("Number of Features")

    plt.show()

def observe_tests():
    images = [torch.load('./GenerationResults/uncond_samples_1.pth'), \
              torch.load('./GenerationResults/uncond_samples_1_40x40.pth')]
    titles = ['32x32', '40x40']

    for n, image in enumerate(images):
        for i in range(len(image)):
            plt.imshow(image[i, 0, ...], cmap='gray')
            plt.title(titles[n])
            plt.show()

if __name__ == "__main__":
    plot_PCA_R2()
    exit(0)
    observe_tests()
    plot_J2()
    plot_PCA()
    compare_statistics()
    large_refinement()
    twentyfiverefined()
    demonstrate_refinement_sigmas()
    demonstrate_closest_patch()
    demonstrate_dataset_generation()
    select_patch_size()
    comp_sbg_to_closest()

