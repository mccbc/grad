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
        url = "http://var.sron.nl/radex/radex.php?action=derive&molfile=" + molfile + "&" + "fmin=" + fmin + "&fmax=" + fmax + "&tbg=2.73&tkin=" + str(t_kin) + "&nh2=" + str(h2_den[i]) + "&cold=" + str(col_den) + "&dv=1.0"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        all_td = soup.find_all('td')
        tex_10_str, tex_21_str = str(all_td[35])[4:-5], str(all_td[40])[4:-5]
        try:
            result[i] = float(tex_10_str), float(tex_21_str)
        except:
            result[i][0] = float(tex_10_str)
    return result

def plot_excitation(col_den, t_kin, molfile='co', crit=True, **kwargs):
    h2_dens = np.logspace(2, 5, 100)
    tex = radex_query(col_den, t_kin, h2_dens, molfile=molfile, **kwargs)
    if molfile=='co':
        label = r'$^{{12}}CO\ $'
        plt.plot(h2_dens, tex[:, 0], 'c-', label=label+'$1-0$')
        plt.plot(h2_dens, tex[:, 1], 'm-', label=label+'$2-1$')
    else:
        label = r'$C^+$'
        plt.plot(h2_dens, tex[:, 0], 'c-', label=label)

    if crit:
        plt.axvline(2572.7, c='k', ls='--', label='$n_{crit, 1-0}$')
        plt.axvline(18185.1, c='gray', ls='--', label='$n_{crit, 2-1}$')
    plt.xlabel(r'$H_2\ [cm^{-3}]$')
    plt.ylabel(r'$T_{ex}\ [K]$')
    plt.xscale('log')
    plt.title('$col\_dens={{{}}}\ cm^{{-2}}, T\_kin={{{}}}\ K$'.format(col_den, t_kin))
    plt.legend()
    plt.tight_layout()
    plt.savefig('coldens={}_Tkin={}.pdf'.format(col_den, t_kin))
    plt.close()


plot_excitation('2.5e17', 10)
plot_excitation('2.5e17', 30)
plot_excitation('2.5e15', 10)
plot_excitation('2.5e15', 30)
plot_excitation('1e16', 100, molfile='c%2B', fmin='1000', fmax='2000', crit=False)
