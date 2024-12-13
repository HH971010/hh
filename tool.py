#一些工具函数
import numpy as np
from matplotlib import pyplot as plt

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(veh_route, ax1, customers, vehicles):
  """
    Plot the vehicle routes on matplotlib axis ax1.
    Args: veh_route (dict): a dictionary of routes keyed by vehicle idx.  ax1
    (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes  customers
    (Customers): the customers instance.  vehicles (Vehicles): the vehicles
    instance.
  """
  veh_used = [v for v in veh_route if veh_route[v] is not None]
  qvs=[]

  cmap = discrete_cmap(vehicles.number + 2, 'nipy_spectral')
  shapes=['o','*','s','v','D']
  s=[5,10,7,10,7]
  colors=['r','sienma','darkorange','darksage','dodgerblue','gold','lime','cyan','purple','blue']
  for veh_number in veh_used:

    lats, lons = zip(*[(c.lat, c.lon) for c in veh_route[veh_number]])
    lats = np.array(lats)
    lons = np.array(lons)
    s_dep = customers.customers[vehicles.starts[veh_number]]
    s_fin = customers.customers[vehicles.ends[veh_number]]

    ax1.plot(lons, lats, marker=shapes[veh_number],markersize=s[veh_number])
    ax1.legend(handles=qvs)
    qv=ax1.quiver(
        lons[:-1],
        lats[:-1],
        lons[1:] - lons[:-1],
        lats[1:] - lats[:-1],
        scale_units='xy',
        angles='xy',
        scale=1,
        width=0.005,
        color='w'


    )
    qvs.append(qv)
    ax1.legend(handles=qvs)