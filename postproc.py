import numpy as np
#from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def read_data(path_output, var,exp):
    X_test = np.load(f'{path_output}/x_test.npy')
    y_test = np.load(f'{path_output}/y_test.npy')
    y_test_predict = np.load(f'{path_output}/y_pred.npy')
    y_test_predict_init = np.load(f'{path_output}/y_pred_init.npy')
    # valloss = np.load(f'./output/{res}/val_loss_{res}_daily_{exp}.npy')
    # trainloss = np.load(f'./output/{res}/train_loss_{res}_daily_{exp}.npy')
    return X_test,y_test,y_test_predict, y_test_predict_init
    # return y_test,y_test_predict,valloss,trainloss

def plot_map(var,X_test,y_test,y_test_predict,y_test_predict_init,exp,metric='mean'):
    if metric == "95th":
        y0 = np.percentile(X_test, 95, axis=0)#*86400*1000
        y1 = np.percentile(y_test, 95, axis=0)#*86400*1000
        y2 = np.percentile(y_test_predict_init, 95, axis=0)#*86400*1000
        y3 = np.percentile(y_test_predict, 95, axis=0)#*86400*1000
    elif metric == "25th":
        y0 = np.percentile(X_test, 25, axis=0)#*86400*1000
        y1 = np.percentile(y_test, 25, axis=0)#*86400*1000
        y2 = np.percentile(y_test_predict_init, 25, axis=0)#*86400*1000
        y3 = np.percentile(y_test_predict, 25, axis=0)#*86400*1000
    else:
        y0 = np.mean(X_test,axis=0)#*86400*1000
        y1 = np.mean(y_test,axis=0)#*86400*1000
        y2 = np.mean(y_test_predict_init,axis = 0)#*86400*1000
        y3 = np.mean(y_test_predict,axis = 0)#*86400*1000

    #print(np.min(y2))
    #print(np.max(y2))
    fig, ax = plt.subplots(2, 2, figsize=(12,8))
    ll = -1
    ul = 1
    if (var == "tmax") or (var == "tmin"):
        cmap = 'Spectral_r'
        ll = np.floor(np.min(y1))
        ul = np.ceil(np.max(y1))
    else:
        # cmap = 'GnBu'
        cmap = 'Spectral'
        if metric == "95th":
            ll = 0
            ul = np.ceil(np.max(y1))
        elif metric == "25th":
            ll = 0
            ul = np.ceil(np.max(y1))
        else:
            ll = 0
            ul = 5
    mm1= ax[0][0].imshow(y0[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[0][0].set_title("Low-Res")
    mm2 = ax[0][1].imshow(y1[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[0][1].set_title("High-Res")
    mm3 = ax[1][0].imshow(y2[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[1][0].set_title("High-Res (Predicted Initial Training)")
    mm4 = ax[1][1].imshow(y3[::-1,:],vmin=ll,vmax=ul,cmap =cmap)
    ax[1][1].set_title("High-Res (Predicted)")
    # plt.colorbar(mm1,ax=ax[0][0],shrink=0.2)
    # plt.colorbar(mm2,ax=ax[0][1],shrink=0.2)
    # plt.colorbar(mm3,ax=ax[1][0],shrink=0.2)
    # plt.colorbar(mm4,ax=ax[1][1],shrink=0.2)
     # Adjust spacing to make room for the colorbar at the bottom
    fig.subplots_adjust(left=0.02, right=.98, top=0.9, bottom=0.10, wspace=0.1)
    # Create a colorbar axis at the bottom and make it bigger
    fig.colorbar(mm1, ax=ax, orientation='horizontal', fraction=0.03)
    # ax[0].remove()
    plt.savefig(f'{path_fig}/spatialmaps_{exp}_{var}_{metric}.pdf')
    # plt.tight_layout()
    plt.close(fig)

def error(y_test,y_test_predict):
    y_testavg = np.mean(y_test[:,:,:],axis=0)
    y_test_predictavg = np.mean(y_test_predict[:,:,:],axis=0)

    mean_sqrd_error = mean_squared_error(y_test.flatten()\
                                         ,y_test_predict.flatten())
    mean_abs_error  = mean_absolute_error(y_test.flatten()\
                                         ,y_test_predict.flatten())
    """mean_sqrd_error = mean_squared_error(y_test.flatten()\
                                         ,y_test_predict.flatten())
    mean_abs_error  = mean_absolute_error(y_test.flatten()\
                                         ,y_test_predict.flatten())"""
    return mean_sqrd_error,mean_abs_error
def plot_diff(var,y_test,y_test_predict,y_test_predict_init,exp,metric='mean'):
    if metric == "95th":
        y1 = np.percentile(y_test, 95, axis=0)#*86400*1000
        y2 = np.percentile(y_test_predict_init, 95, axis=0)#*86400*1000
        y3 = np.percentile(y_test_predict, 95, axis=0)#*86400*1000
    elif metric == "25th":
        y1 = np.percentile(y_test, 25, axis=0)#*86400*1000
        y2 = np.percentile(y_test_predict_init, 25, axis=0)#*86400*1000
        y3 = np.percentile(y_test_predict, 25, axis=0)#*86400*1000
    else:
        y1 = np.mean(y_test,axis=0)#*86400*1000
        y2 = np.mean(y_test_predict_init,axis = 0)#*86400*1000
        y3 = np.mean(y_test_predict,axis = 0)#*86400*1000
    diff1  = y2-y1
    diff2  = y3-y1
    ul = max(np.ceil(np.max(diff1)),np.ceil(np.max(diff2)))
    ll = -ul
    if (var == "tmax") or (var == "tmin"):
        cmap='RdBu_r'
        if var == "tmin":
            ul = 2
            ll = -2
    else:
        cmap='RdBu'
        if metric == "mean":
            ul = 2
            ll = -2
    fig, ax = plt.subplots(1,2,  figsize=(10,4))
    mm1 = ax[0].imshow(diff1[::-1,:],vmin=ll,vmax=ul,cmap=cmap)
    ax[0].set_title("Diff (initial)")
    mm2 = ax[1].imshow(diff2[::-1,:],vmin=ll,vmax=ul,cmap=cmap)
    ax[1].set_title("Diff")

    # plt.colorbar(mm1,ax=ax[0],shrink=0.3)
    # plt.colorbar(mm2,ax=ax[1],shrink=0.3)
    # Add a single color bar at the bottom of both plots
    # Adjust spacing to make room for the colorbar at the bottom
    fig.subplots_adjust(left=0.02, right=.98, top=0.9, bottom=0.10, wspace=0.1)
    # Create a colorbar axis at the bottom and make it bigger
    fig.colorbar(mm1, ax=ax, orientation='horizontal', fraction=0.05)    # cbar.set_label('Colorbar Label')  # Optional: add label to the color bar

    # plt.tight_layout()
    plt.savefig(f'{path_fig}/spatialmaps_{exp}_{var}_{metric}_diff.pdf')

def plot_diff_version(var,exp):
    X_test_1,y_test_1,y_test_predict_1, y_test_predict_init_1 = read_data(path_output, var,exp) 
    X_test_2,y_test_2,y_test_predict_2, y_test_predict_init_2 = read_data(f'./output/{version_diff}', var,exp) 
    y1 = np.mean(y_test_predict_1,axis=0)#*86400*1000
    y2 = np.mean(y_test_predict_2,axis = 0)#*86400*1000
    diff1  = y2-y1
    ul = np.ceil(np.max(diff1))
    ll = -ul
    fig, ax = plt.subplots(1,2,  figsize=(11,8))
    mm1 = ax[0].imshow(diff1[::-1,:],vmin=ll,vmax=ul,cmap='RdBu')
    ax[0].set_title("Diff ")
    ax[1].remove()
    plt.colorbar(mm1,ax=ax[0],shrink=0.3)
    plt.tight_layout()
    plt.savefig(f'{path_fig}/spatialmaps_{exp}_{var}_diff_with_{version_diff}.pdf')


def plot_loss(var,val,train,exp):
    plt.plot(val,label="validation loss")
    plt.plot(train,label="training loss")
    #plt.ylim(0.0014,0.002)
    plt.title(exp)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{path_output}/{var}_val_train_loss_{exp}.pdf')

def plot_density_output(y_test, y_test_predict_init, y_test_predict):
    bins = np.arange(-50, 100 + 5, 5) # 450
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))
    # Plot histogram for the first variable
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
    plt.hist(y_test.flatten(), bins=bins, color='blue', alpha=0.6, edgecolor='black', label="Actual HR Values")
    plt.yscale("log")
    plt.title('Histogram for Actual HR Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Plot histogram for the second variable
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
    plt.hist(y_test_predict_init.flatten(), bins=bins, color='green', alpha=0.6, edgecolor='black', label="Initial Predicted HR Values")
    plt.yscale("log")
    plt.title('Histogram for Initial Predicted HR Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Plot histogram for the third variable
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
    plt.hist(y_test_predict.flatten(), bins=bins, color='red', alpha=0.6, edgecolor='black', label="Predicted HR Values")
    plt.yscale("log")
    plt.title('Histogram for Predicted HR Values')
    plt.xlabel("Value")
    plt.ylabel('Frequency')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'{path_fig}/output_density.png', dpi=300)
    # Show the plot
    plt.tight_layout()
    plt.show()
    return

def main():
    exp = version
    X_test,y_test,y_test_predict, y_test_predict_init = read_data(path_output, var,exp)
    # y_test,y_test_predict,val,train = read_data(var,exp)
    print(np.min(y_test_predict))
    print(np.max(y_test_predict))
    # print(error(y_test,y_test_predict))
    plot_density_output(y_test, y_test_predict_init, y_test_predict)
    if(var == "t2"):
        y_test = y_test - 273.15
        y_test_predict = y_test_predict - 273.15
    plot_map(var,X_test,y_test,y_test_predict,y_test_predict_init,exp)
    plot_map(var,X_test,y_test,y_test_predict,y_test_predict_init,exp, '95th')
    plot_map(var,X_test,y_test,y_test_predict,y_test_predict_init,exp, '25th')
    # plot_loss(var,val,train,exp)
    # # plot_map(var,y_test,y_test_predict,exp)
    plot_diff(var,y_test,y_test_predict, y_test_predict_init,exp)
    plot_diff(var,y_test,y_test_predict, y_test_predict_init,exp, '95th')
    plot_diff(var,y_test,y_test_predict, y_test_predict_init,exp, '25th')
    # plot_diff_version(var,exp)


if __name__ == "__main__":
    version = 'v7.7'
    version_diff = 'v0.1'
    
    var = "tmax"
    path_output = f'./output/{version}'
    path_fig = f'./output/{version}/fig'
    os.makedirs(path_fig, exist_ok=True)

    main()
