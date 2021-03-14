"""Simple taylor diagram
Dalton Kei Sasaki
"""
import numpy as np
import matplotlib.pyplot as plt


def taylor(fig=None,
           ax=None,
           rlocs=None,
           thetamax=np.pi/2,
           stdlim=2,
           grid=False,
           text='Correlation',
           text_dict={'rotation': -45},
           grid_dict={'linestyle': '--', 'linewidth': 0.5}):
    """Taylor diagram axes (polar coordinates)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes._subplots.AxesSubplot
    rlocs : numpy.ndarray
        array with correlation coordinates
        default: [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    thetamax : float
        maximum azimuthal coordinate values (in radians)
        defaut: pi/2
    stdlim : float
        maximum azimuthal coordinate values (in radians)
        defaut: pi/2
    grid : bool
        draws a grid using plt.grid(True, **grid_dict)
        (for more, see grid_dict below)
    text: string
        azimuthal label
        default: 'Correlation'
    text_dict : dict
        adjust the azimuthal label (see above)
    grid_dict : dict
        paramaters of grid (see  above)

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Taylor diagram axes

    """

    # evaluate if figure/ax exists
    if (fig is None) & (ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    elif (fig is not None) & (ax is None):
        ax = fig.add_subplot(111, projection='polar')
    else:
        pass

    # define the correlation coordinates
    if rlocs is None:
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])

    # grid limits
    ax.set_xlim(0, thetamax)
    ax.set_ylim(0, stdlim)

    tlocs = np.arccos(rlocs)        # Conversion to azimuthal angles
    ax.set_thetagrids(tlocs/np.pi*180, labels=rlocs)

    # azimuthal label
    ax.text(np.pi/4,stdlim+stdlim/40,'Correlation', **text_dict)

    # radius label
    ax.set_xlabel('Standard deviation', labelpad=20)

    ax.grid(grid, linestyle='--', linewidth=0.5)
    ax.plot(np.linspace(0, thetamax,100), np.ones(100), c='k', linewidth=0.5)
    return ax



def skill_willmott(re,m):
    """Willmott skill (1981)

    Based on the MSE , a quantitative model skill was presented by Willmott
    (1981). The highest value, WS = 1, means perfect agreement between model
    and observation, while the lowest value,  WS = 0, indicates   complete
    disagreement. This method was used to evaluate ROMS in the simulation
    of multiple parameters in the Hudson River estuary [ Warner et al., 2005b]
    and on the southeast New England Shelf [Wilkin ,2006]. The Willmott skill
    will may be used to  quantify model performance in simulating different
    parameters from the best model run  skill parameter (WILLMOTT, 1981)

    written by: Paula Birocchi

    Parameters
    ----------
    re : numpy.ndarray
        reference data
    m : numpy.ndarray
        evaluated data

    Returns
    -------
    float
        willmott_skill

    """

    dif   = re - m
    soma  = np.nansum(abs(dif)**2)
    somam = m - np.nanmean(re)
    c     = re - np.nanmean(re)
    d     = np.nansum((abs(somam) + abs(c))**2)
    skill = 1 - (soma/d)
    return skill


def skill_willmott_space(rcoef=np.linspace(0,1,100),
                         std=np.linspace(0,2,100),
                         lims = None):
    """Willmott skill as a function of the correlation coefficient and
    standard deviation. This function returns a tuple with input values for
    the plt.contourf method. For instance:

    import matplotlib.pyplot as plt
    plt.contourf(*skill_willmott_space())

    Parameters
    ----------
    rcoef : numpy.ndarray
        array with correlation coefficient values
        default is: np.linspace(0,1,100)
    std : numpy.ndarray
        array with correlation standard deviation values
        default is: np.linspace(0,2,100)
    lims : numpy.ndarray or list
        Contour levels (the function just passes it to the output)
        default is: [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    Returns
    -------
    numpy.ndarray
        xm, ym, zm, lims
        xm, ym: the coordinates of the values in zm
        zm: the height values over which the contour is drawn
        lims:  Determines the number and positions of the contour lines.

    """
    rn = rcoef.size
    sn = std.size
    willmott  = np.zeros([rn,  sn])
    corrcoef  = np.zeros([rn,  sn])
    std_field = np.zeros([rn,  sn])

    if lims  is None:
        lims = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    x = np.arange(0.001,1,0.001)  # synthetic domain
    y0 = np.sin(x*np.pi*10)   # synthetic reference data

    for im, m in enumerate(std):
        for ip, p in enumerate(rcoef):
            y1 = m*np.sin(x*np.pi*10 + np.pi*p)  # syntethic measurement
            willmott[im, ip] = skill_willmott(y0,y1)
            corrcoef[im, ip] = np.arccos(np.corrcoef(y0, y1)[0, 1])
            std0 = np.std(y0)
            std1 = np.std(y1)
            std_field[im, ip] = std1/std0
    return corrcoef, std_field, willmott, lims


def rmse_space(rcoef=np.linspace(0,np.pi/2,100),
               std=np.linspace(0,2,100),
               lims=np.arange(0,10,0.4)):
    """RMSE as a function of the correlation coefficient and
    standard deviation. This function returns a tuple with input values for
    the plt.contourf method. For instance:

    import matplotlib.pyplot as plt
    plt.contourf(*skill_willmott_space())

    Parameters
    ----------
    rcoef : numpy.ndarray
        array with correlation coefficient values
        default is: np.linspace(0,1,100)
    std : numpy.ndarray
        array with correlation standard deviation values
        default is: np.linspace(0,2,100)
    lims : numpy.ndarray or list
        Contour levels (the function just passes it to the output)
        default is: [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    Returns
    -------
    numpy.ndarray
        xm, ym, zm, lims
        xm, ym: the coordinates of the values in zm
        zm: the height values over which the contour is drawn
        lims:  Determines the number and positions of the contour lines.

    """

    xm, ym = np.meshgrid(rcoef, std)
    zm = np.sqrt(1**2+ym**2 - 2*ym*np.cos(xm))
    lims = np.arange(0.2, 10, 0.4)
    return xm, ym, zm, lims


if __name__ == '__main__':
    # -- synthetic data -- #
    phase = np.pi/10  # phase difference between reference and synt_mes
    magnt = 1.5  # magnitude factor
    x = np.arange(0,1,0.001)
    reference = np.sin(x*np.pi*10)  # 'reference'
    synt_mesr = magnt*np.sin(x*np.pi*10+np.pi/10)  # 'measurement'


    # -- taylor diagram -- #
    fig = plt.figure(figsize=[10.2, 3.83])
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122)
    ax1 = taylor(fig, ax=ax1)

    # -- plot lines of skill or rmse -- #
    ax1.contour(*skill_willmott_space(), linewidths=0.5, colors='0.5')
    # ax1.contour(*rmse_space(), linewidths=0.5, colors='0.5')

    # -- convert from cartesian to polar coordinates --
    theta_corr = np.arccos(np.corrcoef(reference, synt_mesr)[0,1])
    std0 = np.std(reference)
    std1 = np.std(synt_mesr)

    # -- divide by the radius standard deviation --
    radius_stdn = std1/std0
    ax1.scatter(theta_corr, radius_stdn)  # plot

    # -- plot time series -- #
    ax2.plot(x, reference, label='reference')
    ax2.plot(x, synt_mesr, label='synthetic')
    ax2.legend()

    plt.tight_layout()
