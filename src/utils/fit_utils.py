import numpy as np
from scipy.optimize import curve_fit

nsigma = 4.0

def SqrtPDF(x, *c):
    m = c[0]
    b = c[1]
    return m*np.sqrt(x)+b

def Pol1PDF(x, *c):
    m = c[0]
    b = c[1]
    return m*x+b

def GausPDF(x, *c):
    #define function parameters
    A = c[0]
    mu = c[1]
    sigma = c[2]
    z = (x-mu)/sigma
    G = A*(1/np.sqrt(2*(sigma**2)*np.pi))*np.exp(-(z**2)/2)
    return G

def sum_nll_gaus(params_, *args):
    _sum = 0.0
    _size = len(args[0])
    params_[0] = 1
    for i in range(_size):
        n = GausPDF(args[0][i], *params_)
        if n!=0: _sum -= np.log(n)
        else: _sum += 1e10
    return _sum


def fit_gaussian_res(earray, ax, dtype, energy):
    
    mean_ = np.mean(earray)
    std_ = np.std(earray)
    mean_ = np.mean(earray[(earray>mean_-20)*(earray<mean_+20)])
    std_ = np.std(earray[(earray>mean_-20)*(earray<mean_+20)])
    
    # plot data
    data_range = (mean_-nsigma*std_, mean_+nsigma*std_)
    M = np.linspace(data_range[0], data_range[1], 100)
   
    data_hist2_array = ax.hist(earray, 
            histtype='step', color='w', linewidth=0,
             range=data_range, bins=50)
    xarray = ((data_hist2_array[1][1:]+data_hist2_array[1][:-1])/2)
    yarray = data_hist2_array[0]
    mask = (xarray>mean_-2.5*std_)*(xarray<mean_+1.0*std_)
    results = curve_fit(GausPDF, xarray[mask], yarray[mask],
                    p0=(max(yarray), mean_, std_),
                    bounds=((0,-1e2,0),(1e3*max(yarray),1e3,1e3)))
    
    data_hist2_array = ax.hist(earray,
             histtype='step', color='w', linewidth=0,
             range=data_radnge, bins=50)
    xarray = ((data_hist2_array[1][1:]+data_hist2_array[1][:-1])/2)
    yarray = data_hist2_array[0]
    
    dE = yarray-np.array([ GausPDF(m, *results[0]) for m in xarray])
    E = (dE/(np.sqrt(yarray+0.001)))
    chi2 = np.sqrt(sum(E[mask]**2)/(len(E[mask])-1))

    ax.scatter(xarray, yarray, marker='o', c='black', s=40)
    ax.bar(xarray, yarray, width=0.05, color='none', yerr=np.sqrt(yarray))
    ax.plot(M, [ GausPDF(m, *results[0]) for m in M], linestyle='dashed')

    # plot data
    ax.set_xlim(data_range)
    ax.set_ylabel('Events (noamlized)', size=14)
    ax.set_xlabel('E-E$_{target}$ (GeV)', size=14)
    stat_text = '''
    A = {:0.2f}
    $\mu$ = {:0.2f}
    $\sigma$ = {:0.2f}
    $\chi^2$ = {:0.2f}
    '''.format(results[0][0], results[0][1], results[0][2], chi2)
    stat_text_box = ax.text(x=mean_+1.2*std_, y=0.7*max(yarray),
        s=stat_text,
        fontsize=12,
        fontfamily='sans-serif',
        horizontalalignment='left', 
        verticalalignment='bottom')
    ax.set_title('E = {} GeV'.format(energy))
    return results[0], dE

def fit_gaussian_data_mc(earray_data, earray_mc, ax, dtype, energy):
    
    mean_ = np.mean(earray_mc)
    mean_ = np.mean(earray_mc[(earray_mc>mean_-20)*(earray_mc<mean_+20)])
    std_ = np.std(earray_mc[(earray_mc>mean_-20)*(earray_mc<mean_+20)])

    data_range = (mean_-nsigma*std_, mean_+nsigma*std_)
    M = np.linspace(data_range[0], data_range[1], 100)

    mc_hist2_array = ax.hist(earray_mc,
             histtype='step', color='w', linewidth=0,
             range=data_range, bins=50)
    xarray = ((mc_hist2_array[1][1:]+mc_hist2_array[1][:-1])/2)
    yarray = mc_hist2_array[0]
    mask = (xarray>mean_-1.0*std_)*(xarray<mean_+2.5*std_)
    results = curve_fit(GausPDF, xarray[mask], yarray[mask],
                    p0=(max(yarray), mean_, std_),
                    bounds=((0,-1e2,0),(1e3*max(yarray),1e3,1e3)))

    ax.scatter(xarray, yarray, marker='o', c='red', s=40)
    ax.bar(xarray, yarray, width=0.05, color='none', yerr=np.sqrt(yarray))
    ax.plot(M, [ GausPDF(m, *results[0]) for m in M], linestyle='-', color='red')
    
    mean_ = np.mean(earray_data)
    mean_ = np.mean(earray_data[(earray_data>mean_-20)*(earray_data<mean_+20)])
    std_ = np.std(earray_data[(earray_data>mean_-20)*(earray_data<mean_+20)])

    data_hist2_array = ax.hist(earray_data,
             histtype='step', color='w', linewidth=0,
             range=data_range, bins=50)
    xarray = ((data_hist2_array[1][1:]+data_hist2_array[1][:-1])/2)
    yarray = data_hist2_array[0]*len(earray_mc)/len(earray_data)
    mask = (xarray>mean_-1.0*std_)*(xarray<mean_+2.5*std_)
    results = curve_fit(GausPDF, xarray[mask], yarray[mask],
                    p0=(max(yarray), mean_, std_),
                    bounds=((0,-1e2,0),(1e3*max(yarray),1e3,1e3)))

    ax.scatter(xarray, yarray, marker='o', c='black', s=40)
    ax.bar(xarray, yarray, width=0.05, color='none', yerr=np.sqrt(yarray))
    ax.plot(M, [ GausPDF(m, *results[0]) for m in M], linestyle='dashed', color='black')

    ax.set_xlim(data_range)
    ax.set_ylabel('Events (noamlized)', size=14)
    ax.set_title('Sum Energy (E = {} GeV)'.format(energy))
    return results[0]

def plot_residual(earray, ax, dtype, energy):
    
    mean_ = np.mean(earray)
    mean_ = np.mean(earray[(earray>mean_-20)*(earray<mean_+20)])
    std_ = np.std(earray[(earray>mean_-20)*(earray<mean_+20)])
    
    # plot data
    data_range = (mean_-20, mean_+20)
    M = np.linspace(data_range[0], data_range[1], 100)
   
    data_hist2_array = ax.hist(earray[(earray>mean_-1.0*std_)*(earray<mean_+2.5*std_)],
             histtype='step', color='w', linewidth=0,
             range=data_range, bins=50)
    xarray = ((data_hist2_array[1][1:]+data_hist2_array[1][:-1])/2)
    yarray = data_hist2_array[0]
    results = curve_fit(GausPDF, xarray, yarray, 
                    p0=(max(yarray), mean_, std_),
                    bounds=((0,-1e2,0),(1e3*max(yarray),1e3,1e3)))
    
    data_hist2_array = ax.hist(earray,
             histtype='step', color='w', linewidth=0,
             range=data_range, bins=50)
    xarray = ((data_hist2_array[1][1:]+data_hist2_array[1][:-1])/2)
    yarray = data_hist2_array[0]
    
    dE = yarray-np.array([ GausPDF(m, *results[0]) for m in xarray])
    E = (dE/(np.sqrt(yarray+0.001)))
    chi2 = np.sqrt(sum(E**2)/(len(E)-1))

    ax.scatter(xarray, yarray, marker='o', c='black', s=40)
    ax.bar(xarray, yarray, width=0.05, color='none', yerr=np.sqrt(yarray))
    ax.plot(M, [ GausPDF(m, *results[0]) for m in M], linestyle='dashed')

    # plot data
    ax.set_xlim(data_range)
    ax.set_ylabel('Events (noamlized)', size=14)
    ax.set_xlabel('E-E$_{target}$ (GeV)', size=14)
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