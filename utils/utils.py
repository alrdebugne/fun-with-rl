import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import display


def render_in_jupyter(
    env, img: mpl.image.AxesImage = None, info: str = ""
) -> mpl.image.AxesImage:
    """ """
    if img is None:
        # Create first image, once
        img = plt.imshow(env.render(mode="rgb_array"))
        img.axes.set_title(info)
        return img

    # Update existing screen
    img.set_data(env.render(mode="rgb_array"))  # just update the data
    img.axes.set_title(info)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    return img
