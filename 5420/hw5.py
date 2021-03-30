from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import pdb

def radex_query(col_den, t_kin, h2_den, molfile='co', fmin='50', fmax='500'):
    result = np.zeros((len(h2_den), 2))
    for i in range(len(h2_den)):
        time.sleep(0.05)
        url = "http://var.sron.nl/radex/radex.php?action=derive&molfile=" + molfile + "&" + "fmin=" + fmin + "&fmax=" + fmax + "&tbg=2.73&tkin=" + str(t_kin) + "&nh2=" + str(h2_den[i]) + "&cold=" + str(col_den).replace('e+', 'e') + "&dv=1.0"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        all_td = soup.find_all('td')
        tex_10_str, tex_21_str = str(all_td[35])[4:-5], str(all_td[40])[4:-5]
        result[i] = float(tex_10_str), float(tex_21_str)
    return result

def plot_excitation(col_den, t_kin, molfile='co', crit=True, **kwargs):
    h2_dens = np.logspace(2, 5, 100)
    tex = radex_query(2.5e17, 10, h2_dens, molfile=molfile, **kwargs)
    if molfile=='co':
        label = r'$^{{12}}CO\ $'
    else:
        label = r'$C^+$'
    plt.plot(h2_dens, tex[:, 0], 'c-', label=label+'$1-0$'.format(molfile.upper()))
    plt.plot(h2_dens, tex[:, 1], 'm-', label=label+'$2-1$'.format(molfile.upper()))
    if crit:
        plt.axvline(x='2572.7', 'y--' label='$n_{crit, 1-0}$')
        plt.axvline(x='18185.1', 'k--' label='$n_{crit, 2-1}$')
    plt.xlabel(r'$H_2\ [cm^{-3}]$')
    plt.ylabel(r'$T_{ex}$')
    plt.xscale('log')
    plt.title('col_dens={}, T_kin={}'.format(col_den, t_kin))
    plt.legend()
    plt.savefig('coldens={}_Tkin={}.pdf'.format(col_den, t_kin))


plot_excitation(2.5e17, 10)
#plot_excitation(2.5e17, 10)
#plot_excitation(2.5e17, 10)
