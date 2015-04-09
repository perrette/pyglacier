import matplotlib.pyplot as plt

#
# plotting
#
def plot_elevation(ds, ax=None):
    if ax is None:
        ax = plt.gca()
    ds['hs'].plot(ax=ax,label="surface")
    ds['hb'].plot(ax=ax,label="bottom")

    # add horizontal line to indicate sea level
    ax.hlines(0, ds.x[0], ds.x[-1], linestyle='dashed', color='black')
    ds['zb'].plot(ax=ax, color='black', linewidth=2, label="bedrock") # add bedrock
    ax.legend(frameon=False, loc="upper right")

def plot_velocity(ds, ax=None):
    if ax is None:
        ax = plt.gca()
    ds = ds.copy()
    u = 'u' if 'u' in ds else 'U'
    ds[u] = ds[u]*3600*24
    ds[u].plot(ax=ax)
    ax.set_ylabel('velocity [m/d]')

def plot_glacier(ds):
    fig,axes=plt.subplots(2,1,sharex=True)
    ax=axes[0]
    plot_elevation(ds, ax)
    ax=axes[1]
    plot_velocity(ds, ax)
    ax.set_xlim([ds.x[0], ds.x[-1]])
    return fig, axes

def plot_stress(ds):
    _v = ["driving", "lat", "long", "basal", "residual"]
    try:
        ds = ds.take(_v)
    except KeyError:
        ds = ds.take([k + '_stress' for k in _v])
    return ds.to_array(axis='stress').T.plot()
