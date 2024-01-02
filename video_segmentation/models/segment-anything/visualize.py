
import torch
import numpy as np
import plotly.graph_objects as go

def get_axes_traces(tform=torch.eye(4), scale=0.3):

    axes = torch.cat([torch.zeros(3,1), torch.tensor([[1,0,0],
                                                      [0,1,0],
                                                      [0,0,1]])], dim=1)
    axes*=scale
    axes = torch.cat([axes, torch.ones(1,4)], dim=0)
    axes = torch.matmul(axes.transpose(0,1), tform)

    axis_start = axes[0]
    axis_x, axis_y, axis_z = axes[1], axes[2], axes[3]
    
    traces = []
    for axis_end,axis_color in zip([axis_x, axis_y, axis_z],
                                   ["red","green","blue"]):
        cone = go.Cone(x=[axis_end[0]],
                      y=[axis_end[1]],
                      z=[axis_end[2]],
                      u=[0.3*(axis_end[0]-axis_start[0])],
                      v=[0.3*(axis_end[1]-axis_start[1])],
                      w=[0.3*(axis_end[2]-axis_start[2])],
                      colorscale=[[0,axis_color],[1,axis_color]],
                      showscale=False)
        traces.append(cone)

        line = go.Scatter3d(x=[axis_start[0],axis_end[0]],
                         y=[axis_start[1],axis_end[1]],
                         z=[axis_start[2],axis_end[2]],
                         mode="lines",
                        line=dict(color=axis_color, width=4),
                         )
        traces.append(line)
        
    return traces

def masks_to_panomasks(masks, colors):
    panomask = np.zeros((480,640,3))
    if len(masks) > 0:
        panomask = np.zeros((masks[0].squeeze().shape[0], masks[0].squeeze().shape[1], 3))

        sorted_indices = np.argsort([int(mask.sum()) for mask in masks])[::-1]

        for sort_idx in sorted_indices:
            panomask[masks[sort_idx]] = colors[sort_idx]
    return panomask
    
def show_masks_with_colors(img, masks, colors, ax=None, alpha=0.35):
    ax_was_none = ax is None
    if ax_was_none:
        fig = plt.figure(figsize=(20,20))
        plt.axis('off')
        ax = plt.gca()
        ax.set_autoscale_on(False)

    ax.imshow(img)

    if len(masks) > 0:
        sorted_indices = np.argsort([int(mask.sum()) for mask in masks])[::-1]
        sorted_anns = sorted(masks, key=(lambda x: x.sum().item()), reverse=True)

        img = np.ones((masks[0].squeeze().shape[0], sorted_anns[0].squeeze().shape[1], 4))
        img[:,:,3] = 0
        for sort_idx in sorted_indices:#sorted_anns:
            color_mask = np.concatenate([colors[sort_idx], [alpha]])
            img[masks[sort_idx]] = color_mask
        ax.imshow(img)
    
    if ax_was_none:
        plt.show()

def show_points(coords, ax, marker_size=150, colors='green'):
    ax.scatter(coords[:, 0], coords[:, 1], color=colors, marker='.', s=marker_size, edgecolor='white', linewidth=0.5)

def show_all_points(samples, colors, img=None, ax=None):
    ax_was_none = ax is None
    if ax_was_none:
        fig = plt.figure(figsize=(20,20))
        plt.axis('off')
        ax = plt.gca()
        
    if img is not None:
        ax.imshow(img)

    for coords, color in zip(samples, colors):
        show_points(coords, ax, colors=color)

    if ax_was_none:
        plt.show()

def get_new_color(colors):
    rand = list(np.random.random(3))
    while rand in colors:
        rand = list(np.random.random(3))
    return rand