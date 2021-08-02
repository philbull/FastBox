
import numpy as np
import pylab as plt
from matplotlib import animation, rc
from IPython.display import HTML

def animate_field(field, coords, coord_fmt="%3.1f MHz", interval=200, 
                  fig=None, colorbar=True):
    """
    Draw an animation that steps through slices of a field. The last dimension 
    of the field is the one that will be stepped through.
    
    Parameters:
        field (array_like):
            3D array of value of a field that can be displayed by 
            `matplotlib.matshow`.
        
        coords (array_like):
            1D array labelling the coordinates that are being stepped through. This 
            will be displayed as a title on each frame of the animation.
        
        coord_fmt (str, optional):
            The format string for the coordinate values.
        
        interval (int, optional):
            Interval between frames, in ms.
        
        fig (matplotlib.Figure, optional):
            Existing Figure to put animation on. Default: None (will create a new 
            Figure).
        
        colorbar (bool, optional):
            Whether to add a `matplotlib.colorbar` to the figure.
        
    Returns:
        animation (HTML5 video):
            Animation as an HTML5 video.
    """
    # Initialise figure
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    
    # Plot initial field and get data ranges
    vmin, vmax = np.min(field), np.max(field)
    im = plt.matshow(field[:,:,0], vmin=vmin, vmax=vmax, 
                     fignum=False)
    cbar = fig.colorbar(im)
    
    def animate(i):
        im.set_data(field[:,:,i])
        ax.set_title(coord_fmt % (coords[i]))
        return (im,)

    def init():
        im.set_data(field[:,:,0])
        ax.set_title(coord_fmt % (coords[0]))
        return (im,)
    
    # Make animation
    anim = animation.FuncAnimation(fig, animate, 
                                   init_func=init,
                                   frames=field.shape[-1], 
                                   interval=interval, 
                                   blit=True)
    return HTML(anim.to_html5_video())
    
