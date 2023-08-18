import numpy as np
from scipy.optimize import curve_fit

def GausPDF(x, A, mu, sigma):
    z = (x-mu)/sigma
    G = A*(1/np.sqrt(2*(sigma**2)*np.pi))*np.exp(-(z**2)/2)
    # G = (1/np.sqrt(2*(sigma**2)*np.pi))*np.exp(-(z**2)/2)
    return G


def line(x, slope, intercept):
    return slope*x + intercept

# def Pol1PDF(x, *c):
#     m = c[0]
#     c = c[1]
#     return m*x+c



def fit_gaussian(earray, ax, dtype, energy, irow, jcol):
    # print("ax ",ax)
    # print('plotting energy = {} GeV in {}, {}'.format(energy, irow, jcol))
    mean_ = np.mean(earray)
    std_ = np.std(earray)
    
    # plot data
    data_range = (mean_-std_*(3), mean_+std_*(3))
    M = np.linspace(data_range[0], data_range[1], 100)
   
    data_hist2_array = ax.hist(earray,
             histtype='step', color='k', linewidth=0.05,
             range=data_range, bins=50)

    xarray = (data_hist2_array[1][1:]+data_hist2_array[1][:-1])/2
    yarray = data_hist2_array[0]
    # temptemp = [xarray,yarray]
    # np.savetxt(f"/home/rusack/vadna042/{e}.csv", np.transpose(temptemp), delimiter=",")

    ax.scatter(xarray, yarray, marker='o', c='black', s=40)
    ax.bar(xarray, yarray, width=0.05, color='none', yerr=np.sqrt(yarray))

    results = curve_fit(GausPDF, xarray, yarray, 
                    p0=(max(yarray), mean_, std_),
                    bounds=((0,0,0),(1e2*max(yarray),1e3,1e3)), maxfev=5000)
    E = yarray-np.array([ GausPDF(m, *results[0]) for m in xarray])
    E = E/(np.sqrt(yarray+0.001))
    chi2 = np.sqrt(sum(E**2)/(len(E)-1))

    ax.plot(M, [ GausPDF(m, *results[0]) for m in M], linestyle='dashed')

    # plot data
    ax.set_xlim(data_range)
    ax.set_ylabel('Events (noamlized)', size=14)
    ax.set_xlabel('Measured Energy (GeV)', size=14)
    stat_text = '''
    A = {:0.2f}
    $\mu$ = {:0.2f}
    $\sigma$ = {:0.2f}
    $\chi^2$ = {:0.2f}
    '''.format(results[0][0], results[0][1], results[0][2], chi2)
    stat_text_box = ax.text(x=mean_+5, y=0.7*max(yarray),
        s=stat_text,
        fontsize=12,
        fontfamily='sans-serif',
        horizontalalignment='left', 
        verticalalignment='bottom')
    ax.set_title('E = {} GeV'.format(energy))
    return results[0]

