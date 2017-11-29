import matplotlib.pyplot as plt
import numpy as np

def triangle_colorbar(ax, dens, border=True, **kwargs):
    '''Create a trianglular colorbar based on RGB values
    ax: axis to plot on
    dens: density of plotting matrix
    border: plot a border on the triangle
    **kwargs: passed onto ax.plot()
    
    Adapted from https://gist.github.com/tboggs/8778945
    '''
    import matplotlib.tri as tri
    
    _corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    _triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
    _midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0 for i in range(3)]
    
    def mask(pt):
        '''Remove points in grid outside of the triangle'''
        x, y = pt
        if x < 0.5:
            if y/x > _corners[2][1]/0.5:
                return 0
            else:
                return 1
        if x > 0.5:
            if y > -1*(_corners[2][1]/0.5)*x + _corners[2][1]/0.5:
                return 0
            else:
                return 1
            
    def xy2bc(xy, tol=1.e-3):
        '''Converts 2D Cartesian coordinates to barycentric.
        Arguments:
            xy: A length-2 sequence containing the x and y value.
        '''
        s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75 for i in range(3)]
        return np.clip(s, tol, 1.0 - tol)
    
    def plot_points(ax, arr, border=True, **kwargs):
        '''Plot a set of grid points colored by RGB sets
        Arguments:
            ax: axis to plot on
            arr: array with dimension Nx5 (X,Y, RGB)
            border (bool): If True, the simplex border is drawn.
            kwargs: Keyword args passed on to ax.plot.
        '''
        ax.scatter(arr[:, 0], arr[:, 1], c=arr[:,2:], s=2, **kwargs)
        ax.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75**0.5)
        ax.axis('off')
        if border is True:
            ax.triplot(_triangle, linewidth=2, c='k')
            
    #Create a grid of points     
    X = np.vstack((np.linspace(0,1,dens),np.linspace(0,1,dens))).T
    xx, yy = np.meshgrid(X,X)
    X = np.dstack((xx,yy)).reshape(xx.shape[0]**2,2)
    del xx, yy
    
    #Remove points outside of the triangle
    maskarr = np.empty(X.shape[0])
    maskarr[:] = map(lambda x: mask(x), X)
    X = X[maskarr == 1]
    del maskarr
    
    #Add color triples to points
    Xb = np.empty((X.shape[0],3))
    Xb[:] = map(lambda x: xy2bc(x), X)
    Xcol = np.hstack((X, Xb))
    del X, Xb
    
    plot_points(ax, Xcol, border, **kwargs)
