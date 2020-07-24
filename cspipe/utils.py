from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import argparse
def setColorConf(colors,ngroups)->list:
    """
    return a list of colors.
    """
    if colors == "hcl":
        try:
            from colorspace import sequential_hcl
            color_repo = sequential_hcl(h=[15,375],l=65,c=70)
            colors_list =  color_repo.colors(ngroups + 1)
        except ImportError:
            print('hcl colorspace package has not being installed.')
            print('please try the following command:')
            print('pip install git+https://github.com/retostauffer/python-colorspace')
    else:
        colors = list(plt.get_cmap(colors).colors)
        colors_list = [to_hex(color) for color in colors]
        colors_list = colors_list[:ngroups]

    return colors_list

def str2bool(v):
    """
    convert str to boolean.
    reference:https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
